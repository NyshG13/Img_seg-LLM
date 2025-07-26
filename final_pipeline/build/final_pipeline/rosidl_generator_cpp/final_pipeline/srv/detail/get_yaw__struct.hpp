// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from final_pipeline:srv/GetYaw.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "final_pipeline/srv/get_yaw.hpp"


#ifndef FINAL_PIPELINE__SRV__DETAIL__GET_YAW__STRUCT_HPP_
#define FINAL_PIPELINE__SRV__DETAIL__GET_YAW__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__final_pipeline__srv__GetYaw_Request __attribute__((deprecated))
#else
# define DEPRECATED__final_pipeline__srv__GetYaw_Request __declspec(deprecated)
#endif

namespace final_pipeline
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetYaw_Request_
{
  using Type = GetYaw_Request_<ContainerAllocator>;

  explicit GetYaw_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit GetYaw_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  // field types and members
  using _structure_needs_at_least_one_member_type =
    uint8_t;
  _structure_needs_at_least_one_member_type structure_needs_at_least_one_member;


  // constant declarations

  // pointer types
  using RawPtr =
    final_pipeline::srv::GetYaw_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const final_pipeline::srv::GetYaw_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<final_pipeline::srv::GetYaw_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<final_pipeline::srv::GetYaw_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      final_pipeline::srv::GetYaw_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<final_pipeline::srv::GetYaw_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      final_pipeline::srv::GetYaw_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<final_pipeline::srv::GetYaw_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<final_pipeline::srv::GetYaw_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<final_pipeline::srv::GetYaw_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__final_pipeline__srv__GetYaw_Request
    std::shared_ptr<final_pipeline::srv::GetYaw_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__final_pipeline__srv__GetYaw_Request
    std::shared_ptr<final_pipeline::srv::GetYaw_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetYaw_Request_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetYaw_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetYaw_Request_

// alias to use template instance with default allocator
using GetYaw_Request =
  final_pipeline::srv::GetYaw_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace final_pipeline


#ifndef _WIN32
# define DEPRECATED__final_pipeline__srv__GetYaw_Response __attribute__((deprecated))
#else
# define DEPRECATED__final_pipeline__srv__GetYaw_Response __declspec(deprecated)
#endif

namespace final_pipeline
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetYaw_Response_
{
  using Type = GetYaw_Response_<ContainerAllocator>;

  explicit GetYaw_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->yaw_deg = 0.0;
    }
  }

  explicit GetYaw_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->yaw_deg = 0.0;
    }
  }

  // field types and members
  using _yaw_deg_type =
    double;
  _yaw_deg_type yaw_deg;

  // setters for named parameter idiom
  Type & set__yaw_deg(
    const double & _arg)
  {
    this->yaw_deg = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    final_pipeline::srv::GetYaw_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const final_pipeline::srv::GetYaw_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<final_pipeline::srv::GetYaw_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<final_pipeline::srv::GetYaw_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      final_pipeline::srv::GetYaw_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<final_pipeline::srv::GetYaw_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      final_pipeline::srv::GetYaw_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<final_pipeline::srv::GetYaw_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<final_pipeline::srv::GetYaw_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<final_pipeline::srv::GetYaw_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__final_pipeline__srv__GetYaw_Response
    std::shared_ptr<final_pipeline::srv::GetYaw_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__final_pipeline__srv__GetYaw_Response
    std::shared_ptr<final_pipeline::srv::GetYaw_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetYaw_Response_ & other) const
  {
    if (this->yaw_deg != other.yaw_deg) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetYaw_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetYaw_Response_

// alias to use template instance with default allocator
using GetYaw_Response =
  final_pipeline::srv::GetYaw_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace final_pipeline


// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__final_pipeline__srv__GetYaw_Event __attribute__((deprecated))
#else
# define DEPRECATED__final_pipeline__srv__GetYaw_Event __declspec(deprecated)
#endif

namespace final_pipeline
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetYaw_Event_
{
  using Type = GetYaw_Event_<ContainerAllocator>;

  explicit GetYaw_Event_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : info(_init)
  {
    (void)_init;
  }

  explicit GetYaw_Event_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : info(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _info_type =
    service_msgs::msg::ServiceEventInfo_<ContainerAllocator>;
  _info_type info;
  using _request_type =
    rosidl_runtime_cpp::BoundedVector<final_pipeline::srv::GetYaw_Request_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<final_pipeline::srv::GetYaw_Request_<ContainerAllocator>>>;
  _request_type request;
  using _response_type =
    rosidl_runtime_cpp::BoundedVector<final_pipeline::srv::GetYaw_Response_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<final_pipeline::srv::GetYaw_Response_<ContainerAllocator>>>;
  _response_type response;

  // setters for named parameter idiom
  Type & set__info(
    const service_msgs::msg::ServiceEventInfo_<ContainerAllocator> & _arg)
  {
    this->info = _arg;
    return *this;
  }
  Type & set__request(
    const rosidl_runtime_cpp::BoundedVector<final_pipeline::srv::GetYaw_Request_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<final_pipeline::srv::GetYaw_Request_<ContainerAllocator>>> & _arg)
  {
    this->request = _arg;
    return *this;
  }
  Type & set__response(
    const rosidl_runtime_cpp::BoundedVector<final_pipeline::srv::GetYaw_Response_<ContainerAllocator>, 1, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<final_pipeline::srv::GetYaw_Response_<ContainerAllocator>>> & _arg)
  {
    this->response = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    final_pipeline::srv::GetYaw_Event_<ContainerAllocator> *;
  using ConstRawPtr =
    const final_pipeline::srv::GetYaw_Event_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<final_pipeline::srv::GetYaw_Event_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<final_pipeline::srv::GetYaw_Event_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      final_pipeline::srv::GetYaw_Event_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<final_pipeline::srv::GetYaw_Event_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      final_pipeline::srv::GetYaw_Event_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<final_pipeline::srv::GetYaw_Event_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<final_pipeline::srv::GetYaw_Event_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<final_pipeline::srv::GetYaw_Event_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__final_pipeline__srv__GetYaw_Event
    std::shared_ptr<final_pipeline::srv::GetYaw_Event_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__final_pipeline__srv__GetYaw_Event
    std::shared_ptr<final_pipeline::srv::GetYaw_Event_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetYaw_Event_ & other) const
  {
    if (this->info != other.info) {
      return false;
    }
    if (this->request != other.request) {
      return false;
    }
    if (this->response != other.response) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetYaw_Event_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetYaw_Event_

// alias to use template instance with default allocator
using GetYaw_Event =
  final_pipeline::srv::GetYaw_Event_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace final_pipeline

namespace final_pipeline
{

namespace srv
{

struct GetYaw
{
  using Request = final_pipeline::srv::GetYaw_Request;
  using Response = final_pipeline::srv::GetYaw_Response;
  using Event = final_pipeline::srv::GetYaw_Event;
};

}  // namespace srv

}  // namespace final_pipeline

#endif  // FINAL_PIPELINE__SRV__DETAIL__GET_YAW__STRUCT_HPP_
