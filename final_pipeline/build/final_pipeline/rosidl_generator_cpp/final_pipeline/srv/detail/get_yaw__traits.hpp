// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from final_pipeline:srv/GetYaw.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "final_pipeline/srv/get_yaw.hpp"


#ifndef FINAL_PIPELINE__SRV__DETAIL__GET_YAW__TRAITS_HPP_
#define FINAL_PIPELINE__SRV__DETAIL__GET_YAW__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "final_pipeline/srv/detail/get_yaw__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace final_pipeline
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetYaw_Request & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetYaw_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetYaw_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace final_pipeline

namespace rosidl_generator_traits
{

[[deprecated("use final_pipeline::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const final_pipeline::srv::GetYaw_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  final_pipeline::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use final_pipeline::srv::to_yaml() instead")]]
inline std::string to_yaml(const final_pipeline::srv::GetYaw_Request & msg)
{
  return final_pipeline::srv::to_yaml(msg);
}

template<>
inline const char * data_type<final_pipeline::srv::GetYaw_Request>()
{
  return "final_pipeline::srv::GetYaw_Request";
}

template<>
inline const char * name<final_pipeline::srv::GetYaw_Request>()
{
  return "final_pipeline/srv/GetYaw_Request";
}

template<>
struct has_fixed_size<final_pipeline::srv::GetYaw_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<final_pipeline::srv::GetYaw_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<final_pipeline::srv::GetYaw_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace final_pipeline
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetYaw_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: yaw_deg
  {
    out << "yaw_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw_deg, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetYaw_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: yaw_deg
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw_deg, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetYaw_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace final_pipeline

namespace rosidl_generator_traits
{

[[deprecated("use final_pipeline::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const final_pipeline::srv::GetYaw_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  final_pipeline::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use final_pipeline::srv::to_yaml() instead")]]
inline std::string to_yaml(const final_pipeline::srv::GetYaw_Response & msg)
{
  return final_pipeline::srv::to_yaml(msg);
}

template<>
inline const char * data_type<final_pipeline::srv::GetYaw_Response>()
{
  return "final_pipeline::srv::GetYaw_Response";
}

template<>
inline const char * name<final_pipeline::srv::GetYaw_Response>()
{
  return "final_pipeline/srv/GetYaw_Response";
}

template<>
struct has_fixed_size<final_pipeline::srv::GetYaw_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<final_pipeline::srv::GetYaw_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<final_pipeline::srv::GetYaw_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace final_pipeline
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetYaw_Event & msg,
  std::ostream & out)
{
  out << "{";
  // member: info
  {
    out << "info: ";
    to_flow_style_yaml(msg.info, out);
    out << ", ";
  }

  // member: request
  {
    if (msg.request.size() == 0) {
      out << "request: []";
    } else {
      out << "request: [";
      size_t pending_items = msg.request.size();
      for (auto item : msg.request) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: response
  {
    if (msg.response.size() == 0) {
      out << "response: []";
    } else {
      out << "response: [";
      size_t pending_items = msg.response.size();
      for (auto item : msg.response) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetYaw_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: info
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "info:\n";
    to_block_style_yaml(msg.info, out, indentation + 2);
  }

  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.request.size() == 0) {
      out << "request: []\n";
    } else {
      out << "request:\n";
      for (auto item : msg.request) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: response
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.response.size() == 0) {
      out << "response: []\n";
    } else {
      out << "response:\n";
      for (auto item : msg.response) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetYaw_Event & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace final_pipeline

namespace rosidl_generator_traits
{

[[deprecated("use final_pipeline::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const final_pipeline::srv::GetYaw_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  final_pipeline::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use final_pipeline::srv::to_yaml() instead")]]
inline std::string to_yaml(const final_pipeline::srv::GetYaw_Event & msg)
{
  return final_pipeline::srv::to_yaml(msg);
}

template<>
inline const char * data_type<final_pipeline::srv::GetYaw_Event>()
{
  return "final_pipeline::srv::GetYaw_Event";
}

template<>
inline const char * name<final_pipeline::srv::GetYaw_Event>()
{
  return "final_pipeline/srv/GetYaw_Event";
}

template<>
struct has_fixed_size<final_pipeline::srv::GetYaw_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<final_pipeline::srv::GetYaw_Event>
  : std::integral_constant<bool, has_bounded_size<final_pipeline::srv::GetYaw_Request>::value && has_bounded_size<final_pipeline::srv::GetYaw_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<final_pipeline::srv::GetYaw_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<final_pipeline::srv::GetYaw>()
{
  return "final_pipeline::srv::GetYaw";
}

template<>
inline const char * name<final_pipeline::srv::GetYaw>()
{
  return "final_pipeline/srv/GetYaw";
}

template<>
struct has_fixed_size<final_pipeline::srv::GetYaw>
  : std::integral_constant<
    bool,
    has_fixed_size<final_pipeline::srv::GetYaw_Request>::value &&
    has_fixed_size<final_pipeline::srv::GetYaw_Response>::value
  >
{
};

template<>
struct has_bounded_size<final_pipeline::srv::GetYaw>
  : std::integral_constant<
    bool,
    has_bounded_size<final_pipeline::srv::GetYaw_Request>::value &&
    has_bounded_size<final_pipeline::srv::GetYaw_Response>::value
  >
{
};

template<>
struct is_service<final_pipeline::srv::GetYaw>
  : std::true_type
{
};

template<>
struct is_service_request<final_pipeline::srv::GetYaw_Request>
  : std::true_type
{
};

template<>
struct is_service_response<final_pipeline::srv::GetYaw_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // FINAL_PIPELINE__SRV__DETAIL__GET_YAW__TRAITS_HPP_
