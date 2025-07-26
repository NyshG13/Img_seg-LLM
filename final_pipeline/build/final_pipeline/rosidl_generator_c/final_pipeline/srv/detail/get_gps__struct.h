// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from final_pipeline:srv/GetGPS.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "final_pipeline/srv/get_gps.h"


#ifndef FINAL_PIPELINE__SRV__DETAIL__GET_GPS__STRUCT_H_
#define FINAL_PIPELINE__SRV__DETAIL__GET_GPS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetGPS in the package final_pipeline.
typedef struct final_pipeline__srv__GetGPS_Request
{
  uint8_t structure_needs_at_least_one_member;
} final_pipeline__srv__GetGPS_Request;

// Struct for a sequence of final_pipeline__srv__GetGPS_Request.
typedef struct final_pipeline__srv__GetGPS_Request__Sequence
{
  final_pipeline__srv__GetGPS_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} final_pipeline__srv__GetGPS_Request__Sequence;

// Constants defined in the message

/// Struct defined in srv/GetGPS in the package final_pipeline.
typedef struct final_pipeline__srv__GetGPS_Response
{
  double latitude;
  double longitude;
} final_pipeline__srv__GetGPS_Response;

// Struct for a sequence of final_pipeline__srv__GetGPS_Response.
typedef struct final_pipeline__srv__GetGPS_Response__Sequence
{
  final_pipeline__srv__GetGPS_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} final_pipeline__srv__GetGPS_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  final_pipeline__srv__GetGPS_Event__request__MAX_SIZE = 1
};
// response
enum
{
  final_pipeline__srv__GetGPS_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/GetGPS in the package final_pipeline.
typedef struct final_pipeline__srv__GetGPS_Event
{
  service_msgs__msg__ServiceEventInfo info;
  final_pipeline__srv__GetGPS_Request__Sequence request;
  final_pipeline__srv__GetGPS_Response__Sequence response;
} final_pipeline__srv__GetGPS_Event;

// Struct for a sequence of final_pipeline__srv__GetGPS_Event.
typedef struct final_pipeline__srv__GetGPS_Event__Sequence
{
  final_pipeline__srv__GetGPS_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} final_pipeline__srv__GetGPS_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FINAL_PIPELINE__SRV__DETAIL__GET_GPS__STRUCT_H_
