/**
 * safety_gate_node.cpp
 * ROS2 node wrapping the C++ SafetyGate
 *
 * Subscriptions:
 *   /arvs/proposed_action   (arvs_msgs/msg/ActionMsg)
 *   /arvs/robot_state       (arvs_msgs/msg/RobotStateMsg)
 *
 * Publications:
 *   /arvs/gate_result       (arvs_msgs/msg/SafetyCheckResultMsg)
 *   /arvs/safety_metrics    (arvs_msgs/msg/SafetyMetricsMsg)
 *
 * Services:
 *   /arvs/emergency_stop    (std_srvs/srv/Trigger)
 *   /arvs/reset_estop       (std_srvs/srv/Trigger)
 *
 * The gate is called synchronously in the action callback.
 * If the gate blocks, the result is published BEFORE returning —
 * the execution controller (another node) waits for this topic.
 */

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

// These message types must be defined in your arvs_msgs package.
// Minimal field definitions are documented inline.
#include <arvs_msgs/msg/action_msg.hpp>
#include <arvs_msgs/msg/robot_state_msg.hpp>
#include <arvs_msgs/msg/safety_check_result_msg.hpp>
#include <arvs_msgs/msg/safety_metrics_msg.hpp>

#include "safety_gate.hpp"
#include "arvs_ros_convert.hpp"   // see conversion utilities below

using namespace arvs;

class SafetyGateNode : public rclcpp::Node {
public:
    explicit SafetyGateNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions())
        : Node("arvs_safety_gate", opts)
    {
        // ── Load safety constraints from parameter server ─────────────────
        SafetyConstraints constraints = load_constraints_from_params();
        gate_ = std::make_unique<SafetyGate>(constraints);

        // ── Subscriptions ─────────────────────────────────────────────────
        sub_action_ = create_subscription<arvs_msgs::msg::ActionMsg>(
            "/arvs/proposed_action", 10,
            std::bind(&SafetyGateNode::on_action, this, std::placeholders::_1));

        sub_state_ = create_subscription<arvs_msgs::msg::RobotStateMsg>(
            "/arvs/robot_state", 10,
            std::bind(&SafetyGateNode::on_state, this, std::placeholders::_1));

        // ── Publications ──────────────────────────────────────────────────
        pub_result_  = create_publisher<arvs_msgs::msg::SafetyCheckResultMsg>(
            "/arvs/gate_result", 10);
        pub_metrics_ = create_publisher<arvs_msgs::msg::SafetyMetricsMsg>(
            "/arvs/safety_metrics", 10);

        // ── Services ──────────────────────────────────────────────────────
        srv_estop_ = create_service<std_srvs::srv::Trigger>(
            "/arvs/emergency_stop",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> res)
            {
                gate_->emergency_stop("operator-commanded via ROS2 service");
                res->success = true;
                res->message = "Emergency stop latched";
                RCLCPP_FATAL(get_logger(), "EMERGENCY STOP activated by operator");
            });

        srv_reset_ = create_service<std_srvs::srv::Trigger>(
            "/arvs/reset_estop",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> res)
            {
                gate_->reset_emergency_stop();
                res->success = true;
                res->message = "Emergency stop reset";
                RCLCPP_WARN(get_logger(), "Emergency stop reset");
            });

        // ── Metrics timer (1 Hz) ──────────────────────────────────────────
        metrics_timer_ = create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&SafetyGateNode::publish_metrics, this));

        RCLCPP_INFO(get_logger(), "ARVS SafetyGateNode ready");
    }

private:
    std::unique_ptr<SafetyGate> gate_;
    RobotState                  latest_state_{};

    rclcpp::Subscription<arvs_msgs::msg::ActionMsg>::SharedPtr      sub_action_;
    rclcpp::Subscription<arvs_msgs::msg::RobotStateMsg>::SharedPtr  sub_state_;
    rclcpp::Publisher<arvs_msgs::msg::SafetyCheckResultMsg>::SharedPtr pub_result_;
    rclcpp::Publisher<arvs_msgs::msg::SafetyMetricsMsg>::SharedPtr     pub_metrics_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_estop_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_reset_;
    rclcpp::TimerBase::SharedPtr metrics_timer_;

    void on_state(const arvs_msgs::msg::RobotStateMsg::SharedPtr msg)
    {
        latest_state_ = ros_to_robot_state(*msg);
    }

    void on_action(const arvs_msgs::msg::ActionMsg::SharedPtr msg)
    {
        const Action action = ros_to_action(*msg);
        const SafetyCheckResult result =
            gate_->check_action(action, latest_state_);

        auto out = safety_result_to_ros(result);
        out.action_id = msg->action_id;
        pub_result_->publish(out);

        if (!result.safe) {
            RCLCPP_WARN(get_logger(),
                "SafetyGate BLOCKED action '%s' (%u violations)",
                action.action_id, result.n_violations);
        }
    }

    void publish_metrics()
    {
        const SafetyGate::Metrics m = gate_->metrics();
        arvs_msgs::msg::SafetyMetricsMsg msg;
        msg.total_checks          = m.total_checks;
        msg.total_blocks          = m.total_blocks;
        msg.last_check_duration_us = m.last_check_duration_us;
        pub_metrics_->publish(msg);
    }

    SafetyConstraints load_constraints_from_params()
    {
        SafetyConstraints c;

        // Joint limits
        declare_parameter("joints.names",       std::vector<std::string>{});
        declare_parameter("joints.max_torque",  std::vector<double>{});
        declare_parameter("joints.max_velocity",std::vector<double>{});

        auto names    = get_parameter("joints.names").as_string_array();
        auto torques  = get_parameter("joints.max_torque").as_double_array();
        auto velocities = get_parameter("joints.max_velocity").as_double_array();

        c.n_joints = static_cast<uint32_t>(
            std::min({names.size(), torques.size(), velocities.size(),
                      static_cast<size_t>(MAX_JOINT_LIMITS)}));

        for (uint32_t i = 0; i < c.n_joints; ++i) {
            std::strncpy(c.joints[i].name, names[i].c_str(), 31);
            c.joints[i].max_torque   = torques[i];
            c.joints[i].max_velocity = velocities[i];
        }

        // Thermal limits
        declare_parameter("thermal.components",      std::vector<std::string>{});
        declare_parameter("thermal.max_temperature", std::vector<double>{});

        auto comp_names = get_parameter("thermal.components").as_string_array();
        auto max_temps  = get_parameter("thermal.max_temperature").as_double_array();

        c.n_thermals = static_cast<uint32_t>(
            std::min({comp_names.size(), max_temps.size(),
                      static_cast<size_t>(MAX_THERMAL_LIMITS)}));

        for (uint32_t i = 0; i < c.n_thermals; ++i) {
            std::strncpy(c.thermals[i].component, comp_names[i].c_str(), 31);
            c.thermals[i].max_temperature = max_temps[i];
        }

        // Battery minimum
        declare_parameter("battery.min_level", MIN_BATTERY_FRACTION);
        c.min_battery = get_parameter("battery.min_level").as_double();

        return c;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SafetyGateNode>());
    rclcpp::shutdown();
    return 0;
}
