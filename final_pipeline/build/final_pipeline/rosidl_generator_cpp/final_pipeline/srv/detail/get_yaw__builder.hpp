// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from final_pipeline:srv/GetYaw.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "final_pipeline/srv/get_yaw.hpp"


#ifndef FINAL_PIPELINE__SRV__DETAIL__GET_YAW__BUILDER_HPP_
#define FINAL_PIPELINE__SRV__DETAIL__GET_YAW__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "final_pipeline/srv/detail/get_yaw__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace final_pipeline
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::final_pipeline::srv::GetYaw_Request>()
{
  return ::final_pipeline::srv::GetYaw_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace final_pipeline


namespace final_pipeline
{

namespace srv
{

namespace builder
{

class Init_GetYaw_Response_yaw_deg
{
public:
  Init_GetYaw_Response_yaw_deg()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::final_pipeline::srv::GetYaw_Response yaw_deg(::final_pipeline::srv::GetYaw_Response::_yaw_deg_type arg)
  {
    msg_.yaw_deg = std::move(arg);
    return std::move(msg_);
  }

private:
  ::final_pipeline::srv::GetYaw_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::final_pipeline::srv::GetYaw_Response>()
{
  return final_pipeline::srv::builder::Init_GetYaw_Response_yaw_deg();
}

}  // namespace final_pipeline


namespace final_pipeline
{

namespace srv
{

namespace builder
{

class Init_GetYaw_Event_response
{
public:
  explicit Init_GetYaw_Event_response(::final_pipeline::srv::GetYaw_Event & msg)
  : msg_(msg)
  {}
  ::final_pipeline::srv::GetYaw_Event response(::final_pipeline::srv::GetYaw_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::final_pipeline::srv::GetYaw_Event msg_;
};

class Init_GetYaw_Event_request
{
public:
  explicit Init_GetYaw_Event_request(::final_pipeline::srv::GetYaw_Event & msg)
  : msg_(msg)
  {}
  Init_GetYaw_Event_response request(::final_pipeline::srv::GetYaw_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_GetYaw_Event_response(msg_);
  }

private:
  ::final_pipeline::srv::GetYaw_Event msg_;
};

class Init_GetYaw_Event_info
{
public:
  Init_GetYaw_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetYaw_Event_request info(::final_pipeline::srv::GetYaw_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_GetYaw_Event_request(msg_);
  }

private:
  ::final_pipeline::srv::GetYaw_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::final_pipeline::srv::GetYaw_Event>()
{
  return final_pipeline::srv::builder::Init_GetYaw_Event_info();
}

}  // namespace final_pipeline

#endif  // FINAL_PIPELINE__SRV__DETAIL__GET_YAW__BUILDER_HPP_
