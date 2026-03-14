/**
 * execution_controller_node.cpp
 * ROS2 node wrapping the C++ ExecutionController
 *
 * Subscriptions:
 *   /arvs/mvi_sequence       (arvs_msgs/msg/MVISequenceMsg)
 *   /arvs/robot_state        (arvs_msgs/msg/RobotStateMsg)
 *   /arvs/gate_result        (arvs_msgs/msg/SafetyCheckResultMsg)
 *   /arvs/authority_valid    (std_msgs/msg/Bool)
 *
 * Publications:
 *   /arvs/sequence_result    (arvs_msgs/msg/SequenceResultMsg)
 *   /arvs/exec_metrics       (arvs_msgs/msg/ExecMetricsMsg)
 *   /arvs/replan_request     (std_msgs/msg/String)  — triggers Python planner
 *
 * Services:
 *   /arvs/stop_execution     (std_srvs/srv/Trigger)
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <arvs_msgs/msg/mvi_sequence_msg.hpp>
#include <arvs_msgs/msg/robot_state_msg.hpp>
#include <arvs_msgs/msg/safety_check_result_msg.hpp>
#include <arvs_msgs/msg/sequence_result_msg.hpp>
#include <arvs_msgs/msg/exec_metrics_msg.hpp>

#include "execution_controller.hpp"
#include "safety_gate.hpp"
#include "watchdog_timer.hpp"
#include "arvs_ros_convert.hpp"

#include <thread>
#include <atomic>

using namespace arvs;

class ExecutionControllerNode : public rclcpp::Node {
public:
    explicit ExecutionControllerNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions())
        : Node("arvs_execution_controller", opts)
    {
        // ── Build components ──────────────────────────────────────────────
        SafetyConstraints constraints = load_constraints_from_params();
        gate_     = std::make_unique<SafetyGate>(constraints);
        watchdog_ = std::make_unique<WatchdogTimer>(
            500,   // 500 ms timeout
            [this](const char* reason) { on_watchdog_timeout(reason); });

        controller_ = std::make_unique<ExecutionController>(
            get_name(), *gate_, *watchdog_, nullptr);  // hw=nullptr until driver attached

        // ── Subscriptions ─────────────────────────────────────────────────
        sub_seq_ = create_subscription<arvs_msgs::msg::MVISequenceMsg>(
            "/arvs/mvi_sequence", 1,
            std::bind(&ExecutionControllerNode::on_sequence, this,
                      std::placeholders::_1));

        sub_state_ = create_subscription<arvs_msgs::msg::RobotStateMsg>(
            "/arvs/robot_state", 10,
            std::bind(&ExecutionControllerNode::on_state, this,
                      std::placeholders::_1));

        sub_auth_ = create_subscription<std_msgs::msg::Bool>(
            "/arvs/authority_valid", rclcpp::QoS(1).reliable(),
            std::bind(&ExecutionControllerNode::on_authority, this,
                      std::placeholders::_1));

        // ── Publications ──────────────────────────────────────────────────
        pub_result_  = create_publisher<arvs_msgs::msg::SequenceResultMsg>(
            "/arvs/sequence_result", 10);
        pub_metrics_ = create_publisher<arvs_msgs::msg::ExecMetricsMsg>(
            "/arvs/exec_metrics", 10);
        pub_replan_  = create_publisher<std_msgs::msg::String>(
            "/arvs/replan_request", 1);

        // ── Services ──────────────────────────────────────────────────────
        srv_stop_ = create_service<std_srvs::srv::Trigger>(
            "/arvs/stop_execution",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> res)
            {
                controller_->request_stop();
                res->success = true;
                res->message = "Stop requested";
            });

        // ── Metrics timer (1 Hz) ──────────────────────────────────────────
        metrics_timer_ = create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&ExecutionControllerNode::publish_metrics, this));

        // ── Start watchdog ────────────────────────────────────────────────
        watchdog_->start();
        watchdog_->pet();

        RCLCPP_INFO(get_logger(), "ARVS ExecutionControllerNode ready");
    }

    ~ExecutionControllerNode() override
    {
        watchdog_->stop();
    }

private:
    std::unique_ptr<SafetyGate>           gate_;
    std::unique_ptr<WatchdogTimer>        watchdog_;
    std::unique_ptr<ExecutionController>  controller_;

    RobotState          latest_state_{};
    std::atomic<bool>   authority_valid_{false};
    std::thread         exec_thread_;

    rclcpp::Subscription<arvs_msgs::msg::MVISequenceMsg>::SharedPtr  sub_seq_;
    rclcpp::Subscription<arvs_msgs::msg::RobotStateMsg>::SharedPtr   sub_state_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr              sub_auth_;

    rclcpp::Publisher<arvs_msgs::msg::SequenceResultMsg>::SharedPtr  pub_result_;
    rclcpp::Publisher<arvs_msgs::msg::ExecMetricsMsg>::SharedPtr     pub_metrics_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr               pub_replan_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr                srv_stop_;
    rclcpp::TimerBase::SharedPtr                                      metrics_timer_;

    // ── Callbacks ─────────────────────────────────────────────────────────

    void on_state(const arvs_msgs::msg::RobotStateMsg::SharedPtr msg)
    {
        latest_state_ = ros_to_robot_state(*msg);
        watchdog_->pet();  // receiving state = proof the pipeline is alive
    }

    void on_authority(const std_msgs::msg::Bool::SharedPtr msg)
    {
        authority_valid_.store(msg->data);
        if (!msg->data) {
            RCLCPP_ERROR(get_logger(),
                "authority_valid=false received — stopping execution");
            controller_->request_stop();
        }
    }

    void on_sequence(const arvs_msgs::msg::MVISequenceMsg::SharedPtr msg)
    {
        if (!authority_valid_.load()) {
            RCLCPP_WARN(get_logger(),
                "Ignoring incoming sequence '%s' — authority_valid=false",
                msg->sequence_id.c_str());
            return;
        }

        if (controller_->is_executing()) {
            RCLCPP_WARN(get_logger(),
                "Already executing; ignoring new sequence '%s'",
                msg->sequence_id.c_str());
            return;
        }

        const MVISequence seq = ros_to_mvi_sequence(*msg);

        // Run execution in a separate thread to not block the ROS2 executor
        if (exec_thread_.joinable()) exec_thread_.join();
        exec_thread_ = std::thread([this, seq]() {
            const SequenceResult result =
                controller_->execute_sequence(seq, latest_state_);

            pub_result_->publish(sequence_result_to_ros(result));

            // If partial success or failure, request replanning
            if (result.status == ExecStatus::PARTIAL_SUCCESS ||
                result.status == ExecStatus::FAILED) {
                std_msgs::msg::String rp;
                rp.data = seq.sequence_id;
                pub_replan_->publish(rp);
                RCLCPP_WARN(get_logger(),
                    "Replan requested for sequence '%s'", seq.sequence_id);
            }
        });
    }

    void publish_metrics()
    {
        const ExecutionController::Metrics m = controller_->metrics();
        arvs_msgs::msg::ExecMetricsMsg msg;
        msg.sequences_run        = m.sequences_run;
        msg.actions_run          = m.actions_run;
        msg.gate_blocks          = m.gate_blocks;
        msg.emergency_stops      = m.emergency_stops;
        msg.replan_requests      = m.replan_requests;
        msg.avg_position_error_m = m.avg_position_error_m;
        pub_metrics_->publish(msg);
    }

    void on_watchdog_timeout(const char* reason)
    {
        RCLCPP_FATAL(get_logger(), "WATCHDOG TIMEOUT: %s", reason);
        gate_->emergency_stop(reason);
        controller_->request_stop();
        // The WatchdogTimer will send SIGTERM after this callback returns
    }

    SafetyConstraints load_constraints_from_params()
    {
        SafetyConstraints c;
        // Same parameter loading as SafetyGateNode — extract to shared util if needed
        declare_parameter("joints.names",       std::vector<std::string>{});
        declare_parameter("joints.max_torque",  std::vector<double>{});
        auto names   = get_parameter("joints.names").as_string_array();
        auto torques = get_parameter("joints.max_torque").as_double_array();
        c.n_joints   = std::min(names.size(), torques.size());
        for (uint32_t i = 0; i < c.n_joints; ++i) {
            std::strncpy(c.joints[i].name, names[i].c_str(), 31);
            c.joints[i].max_torque = torques[i];
        }
        declare_parameter("battery.min_level", MIN_BATTERY_FRACTION);
        c.min_battery = get_parameter("battery.min_level").as_double();
        return c;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ExecutionControllerNode>());
    rclcpp::shutdown();
    return 0;
}
