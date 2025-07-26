// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from final_pipeline:srv/GetGPS.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "final_pipeline/srv/get_gps.hpp"


#ifndef FINAL_PIPELINE__SRV__DETAIL__GET_GPS__BUILDER_HPP_
#define FINAL_PIPELINE__SRV__DETAIL__GET_GPS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "final_pipeline/srv/detail/get_gps__struct.hpp"
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
auto build<::final_pipeline::srv::GetGPS_Request>()
{
  return ::final_pipeline::srv::GetGPS_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace final_pipeline


namespace final_pipeline
{

namespace srv
{

namespace builder
{

class Init_GetGPS_Response_longitude
{
public:
  explicit Init_GetGPS_Response_longitude(::final_pipeline::srv::GetGPS_Response & msg)
  : msg_(msg)
  {}
  ::final_pipeline::srv::GetGPS_Response longitude(::final_pipeline::srv::GetGPS_Response::_longitude_type arg)
  {
    msg_.longitude = std::move(arg);
    return std::move(msg_);
  }

private:
  ::final_pipeline::srv::GetGPS_Response msg_;
};

class Init_GetGPS_Response_latitude
{
public:
  Init_GetGPS_Response_latitude()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetGPS_Response_longitude latitude(::final_pipeline::srv::GetGPS_Response::_latitude_type arg)
  {
    msg_.latitude = std::move(arg);
    return Init_GetGPS_Response_longitude(msg_);
  }

private:
  ::final_pipeline::srv::GetGPS_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::final_pipeline::srv::GetGPS_Response>()
{
  return final_pipeline::srv::builder::Init_GetGPS_Response_latitude();
}

}  // namespace final_pipeline


namespace final_pipeline
{

namespace srv
{

namespace builder
{

class Init_GetGPS_Event_response
{
public:
  explicit Init_GetGPS_Event_response(::final_pipeline::srv::GetGPS_Event & msg)
  : msg_(msg)
  {}
  ::final_pipeline::srv::GetGPS_Event response(::final_pipeline::srv::GetGPS_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::final_pipeline::srv::GetGPS_Event msg_;
};

class Init_GetGPS_Event_request
{
public:
  explicit Init_GetGPS_Event_request(::final_pipeline::srv::GetGPS_Event & msg)
  : msg_(msg)
  {}
  Init_GetGPS_Event_response request(::final_pipeline::srv::GetGPS_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_GetGPS_Event_response(msg_);
  }

private:
  ::final_pipeline::srv::GetGPS_Event msg_;
};

class Init_GetGPS_Event_info
{
public:
  Init_GetGPS_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetGPS_Event_request info(::final_pipeline::srv::GetGPS_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_GetGPS_Event_request(msg_);
  }

private:
  ::final_pipeline::srv::GetGPS_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::final_pipeline::srv::GetGPS_Event>()
{
  return final_pipeline::srv::builder::Init_GetGPS_Event_info();
}

}  // namespace final_pipeline

#endif  // FINAL_PIPELINE__SRV__DETAIL__GET_GPS__BUILDER_HPP_
