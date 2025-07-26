// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from final_pipeline:srv/GetDepth.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "final_pipeline/srv/get_depth.hpp"


#ifndef FINAL_PIPELINE__SRV__DETAIL__GET_DEPTH__BUILDER_HPP_
#define FINAL_PIPELINE__SRV__DETAIL__GET_DEPTH__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "final_pipeline/srv/detail/get_depth__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace final_pipeline
{

namespace srv
{

namespace builder
{

class Init_GetDepth_Request_pixel_y
{
public:
  explicit Init_GetDepth_Request_pixel_y(::final_pipeline::srv::GetDepth_Request & msg)
  : msg_(msg)
  {}
  ::final_pipeline::srv::GetDepth_Request pixel_y(::final_pipeline::srv::GetDepth_Request::_pixel_y_type arg)
  {
    msg_.pixel_y = std::move(arg);
    return std::move(msg_);
  }

private:
  ::final_pipeline::srv::GetDepth_Request msg_;
};

class Init_GetDepth_Request_pixel_x
{
public:
  Init_GetDepth_Request_pixel_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetDepth_Request_pixel_y pixel_x(::final_pipeline::srv::GetDepth_Request::_pixel_x_type arg)
  {
    msg_.pixel_x = std::move(arg);
    return Init_GetDepth_Request_pixel_y(msg_);
  }

private:
  ::final_pipeline::srv::GetDepth_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::final_pipeline::srv::GetDepth_Request>()
{
  return final_pipeline::srv::builder::Init_GetDepth_Request_pixel_x();
}

}  // namespace final_pipeline


namespace final_pipeline
{

namespace srv
{

namespace builder
{

class Init_GetDepth_Response_depth
{
public:
  Init_GetDepth_Response_depth()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::final_pipeline::srv::GetDepth_Response depth(::final_pipeline::srv::GetDepth_Response::_depth_type arg)
  {
    msg_.depth = std::move(arg);
    return std::move(msg_);
  }

private:
  ::final_pipeline::srv::GetDepth_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::final_pipeline::srv::GetDepth_Response>()
{
  return final_pipeline::srv::builder::Init_GetDepth_Response_depth();
}

}  // namespace final_pipeline


namespace final_pipeline
{

namespace srv
{

namespace builder
{

class Init_GetDepth_Event_response
{
public:
  explicit Init_GetDepth_Event_response(::final_pipeline::srv::GetDepth_Event & msg)
  : msg_(msg)
  {}
  ::final_pipeline::srv::GetDepth_Event response(::final_pipeline::srv::GetDepth_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::final_pipeline::srv::GetDepth_Event msg_;
};

class Init_GetDepth_Event_request
{
public:
  explicit Init_GetDepth_Event_request(::final_pipeline::srv::GetDepth_Event & msg)
  : msg_(msg)
  {}
  Init_GetDepth_Event_response request(::final_pipeline::srv::GetDepth_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_GetDepth_Event_response(msg_);
  }

private:
  ::final_pipeline::srv::GetDepth_Event msg_;
};

class Init_GetDepth_Event_info
{
public:
  Init_GetDepth_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetDepth_Event_request info(::final_pipeline::srv::GetDepth_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_GetDepth_Event_request(msg_);
  }

private:
  ::final_pipeline::srv::GetDepth_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::final_pipeline::srv::GetDepth_Event>()
{
  return final_pipeline::srv::builder::Init_GetDepth_Event_info();
}

}  // namespace final_pipeline

#endif  // FINAL_PIPELINE__SRV__DETAIL__GET_DEPTH__BUILDER_HPP_
