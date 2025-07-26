// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from final_pipeline:srv/GetYaw.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "final_pipeline/srv/detail/get_yaw__rosidl_typesupport_introspection_c.h"
#include "final_pipeline/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "final_pipeline/srv/detail/get_yaw__functions.h"
#include "final_pipeline/srv/detail/get_yaw__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  final_pipeline__srv__GetYaw_Request__init(message_memory);
}

void final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_fini_function(void * message_memory)
{
  final_pipeline__srv__GetYaw_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_message_member_array[1] = {
  {
    "structure_needs_at_least_one_member",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(final_pipeline__srv__GetYaw_Request, structure_needs_at_least_one_member),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_message_members = {
  "final_pipeline__srv",  // message namespace
  "GetYaw_Request",  // message name
  1,  // number of fields
  sizeof(final_pipeline__srv__GetYaw_Request),
  false,  // has_any_key_member_
  final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_message_member_array,  // message members
  final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_message_type_support_handle = {
  0,
  &final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_message_members,
  get_message_typesupport_handle_function,
  &final_pipeline__srv__GetYaw_Request__get_type_hash,
  &final_pipeline__srv__GetYaw_Request__get_type_description,
  &final_pipeline__srv__GetYaw_Request__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_final_pipeline
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw_Request)() {
  if (!final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_message_type_support_handle.typesupport_identifier) {
    final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "final_pipeline/srv/detail/get_yaw__rosidl_typesupport_introspection_c.h"
// already included above
// #include "final_pipeline/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "final_pipeline/srv/detail/get_yaw__functions.h"
// already included above
// #include "final_pipeline/srv/detail/get_yaw__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  final_pipeline__srv__GetYaw_Response__init(message_memory);
}

void final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_fini_function(void * message_memory)
{
  final_pipeline__srv__GetYaw_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_message_member_array[1] = {
  {
    "yaw_deg",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(final_pipeline__srv__GetYaw_Response, yaw_deg),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_message_members = {
  "final_pipeline__srv",  // message namespace
  "GetYaw_Response",  // message name
  1,  // number of fields
  sizeof(final_pipeline__srv__GetYaw_Response),
  false,  // has_any_key_member_
  final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_message_member_array,  // message members
  final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_message_type_support_handle = {
  0,
  &final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_message_members,
  get_message_typesupport_handle_function,
  &final_pipeline__srv__GetYaw_Response__get_type_hash,
  &final_pipeline__srv__GetYaw_Response__get_type_description,
  &final_pipeline__srv__GetYaw_Response__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_final_pipeline
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw_Response)() {
  if (!final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_message_type_support_handle.typesupport_identifier) {
    final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "final_pipeline/srv/detail/get_yaw__rosidl_typesupport_introspection_c.h"
// already included above
// #include "final_pipeline/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "final_pipeline/srv/detail/get_yaw__functions.h"
// already included above
// #include "final_pipeline/srv/detail/get_yaw__struct.h"


// Include directives for member types
// Member `info`
#include "service_msgs/msg/service_event_info.h"
// Member `info`
#include "service_msgs/msg/detail/service_event_info__rosidl_typesupport_introspection_c.h"
// Member `request`
// Member `response`
#include "final_pipeline/srv/get_yaw.h"
// Member `request`
// Member `response`
// already included above
// #include "final_pipeline/srv/detail/get_yaw__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  final_pipeline__srv__GetYaw_Event__init(message_memory);
}

void final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_fini_function(void * message_memory)
{
  final_pipeline__srv__GetYaw_Event__fini(message_memory);
}

size_t final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__size_function__GetYaw_Event__request(
  const void * untyped_member)
{
  const final_pipeline__srv__GetYaw_Request__Sequence * member =
    (const final_pipeline__srv__GetYaw_Request__Sequence *)(untyped_member);
  return member->size;
}

const void * final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_const_function__GetYaw_Event__request(
  const void * untyped_member, size_t index)
{
  const final_pipeline__srv__GetYaw_Request__Sequence * member =
    (const final_pipeline__srv__GetYaw_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void * final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_function__GetYaw_Event__request(
  void * untyped_member, size_t index)
{
  final_pipeline__srv__GetYaw_Request__Sequence * member =
    (final_pipeline__srv__GetYaw_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__fetch_function__GetYaw_Event__request(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const final_pipeline__srv__GetYaw_Request * item =
    ((const final_pipeline__srv__GetYaw_Request *)
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_const_function__GetYaw_Event__request(untyped_member, index));
  final_pipeline__srv__GetYaw_Request * value =
    (final_pipeline__srv__GetYaw_Request *)(untyped_value);
  *value = *item;
}

void final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__assign_function__GetYaw_Event__request(
  void * untyped_member, size_t index, const void * untyped_value)
{
  final_pipeline__srv__GetYaw_Request * item =
    ((final_pipeline__srv__GetYaw_Request *)
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_function__GetYaw_Event__request(untyped_member, index));
  const final_pipeline__srv__GetYaw_Request * value =
    (const final_pipeline__srv__GetYaw_Request *)(untyped_value);
  *item = *value;
}

bool final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__resize_function__GetYaw_Event__request(
  void * untyped_member, size_t size)
{
  final_pipeline__srv__GetYaw_Request__Sequence * member =
    (final_pipeline__srv__GetYaw_Request__Sequence *)(untyped_member);
  final_pipeline__srv__GetYaw_Request__Sequence__fini(member);
  return final_pipeline__srv__GetYaw_Request__Sequence__init(member, size);
}

size_t final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__size_function__GetYaw_Event__response(
  const void * untyped_member)
{
  const final_pipeline__srv__GetYaw_Response__Sequence * member =
    (const final_pipeline__srv__GetYaw_Response__Sequence *)(untyped_member);
  return member->size;
}

const void * final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_const_function__GetYaw_Event__response(
  const void * untyped_member, size_t index)
{
  const final_pipeline__srv__GetYaw_Response__Sequence * member =
    (const final_pipeline__srv__GetYaw_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void * final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_function__GetYaw_Event__response(
  void * untyped_member, size_t index)
{
  final_pipeline__srv__GetYaw_Response__Sequence * member =
    (final_pipeline__srv__GetYaw_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__fetch_function__GetYaw_Event__response(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const final_pipeline__srv__GetYaw_Response * item =
    ((const final_pipeline__srv__GetYaw_Response *)
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_const_function__GetYaw_Event__response(untyped_member, index));
  final_pipeline__srv__GetYaw_Response * value =
    (final_pipeline__srv__GetYaw_Response *)(untyped_value);
  *value = *item;
}

void final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__assign_function__GetYaw_Event__response(
  void * untyped_member, size_t index, const void * untyped_value)
{
  final_pipeline__srv__GetYaw_Response * item =
    ((final_pipeline__srv__GetYaw_Response *)
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_function__GetYaw_Event__response(untyped_member, index));
  const final_pipeline__srv__GetYaw_Response * value =
    (const final_pipeline__srv__GetYaw_Response *)(untyped_value);
  *item = *value;
}

bool final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__resize_function__GetYaw_Event__response(
  void * untyped_member, size_t size)
{
  final_pipeline__srv__GetYaw_Response__Sequence * member =
    (final_pipeline__srv__GetYaw_Response__Sequence *)(untyped_member);
  final_pipeline__srv__GetYaw_Response__Sequence__fini(member);
  return final_pipeline__srv__GetYaw_Response__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_member_array[3] = {
  {
    "info",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(final_pipeline__srv__GetYaw_Event, info),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "request",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(final_pipeline__srv__GetYaw_Event, request),  // bytes offset in struct
    NULL,  // default value
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__size_function__GetYaw_Event__request,  // size() function pointer
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_const_function__GetYaw_Event__request,  // get_const(index) function pointer
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_function__GetYaw_Event__request,  // get(index) function pointer
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__fetch_function__GetYaw_Event__request,  // fetch(index, &value) function pointer
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__assign_function__GetYaw_Event__request,  // assign(index, value) function pointer
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__resize_function__GetYaw_Event__request  // resize(index) function pointer
  },
  {
    "response",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(final_pipeline__srv__GetYaw_Event, response),  // bytes offset in struct
    NULL,  // default value
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__size_function__GetYaw_Event__response,  // size() function pointer
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_const_function__GetYaw_Event__response,  // get_const(index) function pointer
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__get_function__GetYaw_Event__response,  // get(index) function pointer
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__fetch_function__GetYaw_Event__response,  // fetch(index, &value) function pointer
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__assign_function__GetYaw_Event__response,  // assign(index, value) function pointer
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__resize_function__GetYaw_Event__response  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_members = {
  "final_pipeline__srv",  // message namespace
  "GetYaw_Event",  // message name
  3,  // number of fields
  sizeof(final_pipeline__srv__GetYaw_Event),
  false,  // has_any_key_member_
  final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_member_array,  // message members
  final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_init_function,  // function to initialize message memory (memory has to be allocated)
  final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_type_support_handle = {
  0,
  &final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_members,
  get_message_typesupport_handle_function,
  &final_pipeline__srv__GetYaw_Event__get_type_hash,
  &final_pipeline__srv__GetYaw_Event__get_type_description,
  &final_pipeline__srv__GetYaw_Event__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_final_pipeline
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw_Event)() {
  final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, service_msgs, msg, ServiceEventInfo)();
  final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw_Request)();
  final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw_Response)();
  if (!final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_type_support_handle.typesupport_identifier) {
    final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "final_pipeline/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "final_pipeline/srv/detail/get_yaw__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers final_pipeline__srv__detail__get_yaw__rosidl_typesupport_introspection_c__GetYaw_service_members = {
  "final_pipeline__srv",  // service namespace
  "GetYaw",  // service name
  // the following fields are initialized below on first access
  NULL,  // request message
  // final_pipeline__srv__detail__get_yaw__rosidl_typesupport_introspection_c__GetYaw_Request_message_type_support_handle,
  NULL,  // response message
  // final_pipeline__srv__detail__get_yaw__rosidl_typesupport_introspection_c__GetYaw_Response_message_type_support_handle
  NULL  // event_message
  // final_pipeline__srv__detail__get_yaw__rosidl_typesupport_introspection_c__GetYaw_Response_message_type_support_handle
};


static rosidl_service_type_support_t final_pipeline__srv__detail__get_yaw__rosidl_typesupport_introspection_c__GetYaw_service_type_support_handle = {
  0,
  &final_pipeline__srv__detail__get_yaw__rosidl_typesupport_introspection_c__GetYaw_service_members,
  get_service_typesupport_handle_function,
  &final_pipeline__srv__GetYaw_Request__rosidl_typesupport_introspection_c__GetYaw_Request_message_type_support_handle,
  &final_pipeline__srv__GetYaw_Response__rosidl_typesupport_introspection_c__GetYaw_Response_message_type_support_handle,
  &final_pipeline__srv__GetYaw_Event__rosidl_typesupport_introspection_c__GetYaw_Event_message_type_support_handle,
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_CREATE_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    final_pipeline,
    srv,
    GetYaw
  ),
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_DESTROY_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    final_pipeline,
    srv,
    GetYaw
  ),
  &final_pipeline__srv__GetYaw__get_type_hash,
  &final_pipeline__srv__GetYaw__get_type_description,
  &final_pipeline__srv__GetYaw__get_type_description_sources,
};

// Forward declaration of message type support functions for service members
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw_Request)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw_Response)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw_Event)(void);

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_final_pipeline
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw)(void) {
  if (!final_pipeline__srv__detail__get_yaw__rosidl_typesupport_introspection_c__GetYaw_service_type_support_handle.typesupport_identifier) {
    final_pipeline__srv__detail__get_yaw__rosidl_typesupport_introspection_c__GetYaw_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)final_pipeline__srv__detail__get_yaw__rosidl_typesupport_introspection_c__GetYaw_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw_Response)()->data;
  }
  if (!service_members->event_members_) {
    service_members->event_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, final_pipeline, srv, GetYaw_Event)()->data;
  }

  return &final_pipeline__srv__detail__get_yaw__rosidl_typesupport_introspection_c__GetYaw_service_type_support_handle;
}
