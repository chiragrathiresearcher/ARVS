/**
 * axiom_validator_node.cpp
 * ROS2 node wrapping the C++ AxiomValidator
 *
 * Subscriptions:
 *   /arvs/axiom_system_state   (arvs_msgs/msg/AxiomSystemStateMsg) — 10 Hz
 *
 * Publications:
 *   /arvs/axiom_result         (arvs_msgs/msg/AxiomValidationResultMsg)
 *   /arvs/authority_valid      (std_msgs/msg/Bool)  — gating signal
 *
 * Services:
 *   /arvs/validate_axiom       (arvs_msgs/srv/ValidateAxiom)
 *
 * On a CRITICAL violation (authority_valid = false), this node
 * calls /arvs/emergency_stop via a ROS2 service client.
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <arvs_msgs/msg/axiom_system_state_msg.hpp>
#include <arvs_msgs/msg/axiom_validation_result_msg.hpp>

#include "axiom_validator.hpp"
#include "arvs_ros_convert.hpp"

using namespace arvs;

class AxiomValidatorNode : public rclcpp::Node {
public:
    explicit AxiomValidatorNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions())
        : Node("arvs_axiom_validator", opts)
    {
        // ── Subscription ─────────────────────────────────────────────────
        sub_ = create_subscription<arvs_msgs::msg::AxiomSystemStateMsg>(
            "/arvs/axiom_system_state", 10,
            std::bind(&AxiomValidatorNode::on_state, this, std::placeholders::_1));

        // ── Publications ──────────────────────────────────────────────────
        pub_result_    = create_publisher<arvs_msgs::msg::AxiomValidationResultMsg>(
            "/arvs/axiom_result", 10);
        pub_authority_ = create_publisher<std_msgs::msg::Bool>(
            "/arvs/authority_valid", rclcpp::QoS(1).reliable());

        // ── Emergency stop client ─────────────────────────────────────────
        estop_client_ = create_client<std_srvs::srv::Trigger>("/arvs/emergency_stop");

        RCLCPP_INFO(get_logger(), "ARVS AxiomValidatorNode ready");
    }

private:
    AxiomValidator validator_;

    rclcpp::Subscription<arvs_msgs::msg::AxiomSystemStateMsg>::SharedPtr sub_;
    rclcpp::Publisher<arvs_msgs::msg::AxiomValidationResultMsg>::SharedPtr pub_result_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_authority_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr estop_client_;

    void on_state(const arvs_msgs::msg::AxiomSystemStateMsg::SharedPtr msg)
    {
        const AxiomSystemState state   = ros_to_axiom_state(*msg);
        const double           now_sec = now().seconds();

        const AxiomValidationResult result = validator_.validate(state, now_sec);

        // ── Publish full validation result ───────────────────────────────
        pub_result_->publish(axiom_result_to_ros(result));

        // ── Publish authority signal (Boolean gating) ────────────────────
        std_msgs::msg::Bool auth_msg;
        auth_msg.data = result.authority_valid;
        pub_authority_->publish(auth_msg);

        // ── Escalate critical failures ───────────────────────────────────
        if (!result.authority_valid) {
            RCLCPP_ERROR(get_logger(),
                "Axiom validation FAILED — authority_valid=false. "
                "action_permitted=%d. Triggering emergency stop.",
                (int)result.action_permitted);

            trigger_emergency_stop("axiom_validator: authority_valid=false");
        }
    }

    void trigger_emergency_stop(const char* reason)
    {
        if (!estop_client_->service_is_ready()) {
            RCLCPP_FATAL(get_logger(),
                "Emergency stop service not ready! Cannot escalate: %s", reason);
            return;
        }
        auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
        // Fire-and-forget: we don't wait for response in the hot path
        estop_client_->async_send_request(req,
            [this, reason_str = std::string(reason)](
                rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture fut)
            {
                auto resp = fut.get();
                if (!resp->success) {
                    RCLCPP_FATAL(get_logger(),
                        "Emergency stop service returned failure for: %s",
                        reason_str.c_str());
                }
            });
    }
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AxiomValidatorNode>());
    rclcpp::shutdown();
    return 0;
}
