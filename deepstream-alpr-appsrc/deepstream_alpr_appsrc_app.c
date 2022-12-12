/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"

#define CONFIG_PATH "deepstream_alpr_appsrc_app_config.txt"

#define PGIE_CONFIG_FILE "alpr_pgie_config.txt"
#define SGIE0_CONFIG_FILE "alpr_sgie0_config.txt"
#define SGIE1_CONFIG_FILE "alpr_sgie1_config.txt"
#define SGIE2_CONFIG_FILE "alpr_sgie2_config.txt"
#define SGIE3_CONFIG_FILE "alpr_sgie3_config.txt"
#define SGIE4_CONFIG_FILE "alpr_sgie4_config.txt"
#define MAX_DISPLAY_LEN 64

#define TRACKER_CONFIG_FILE "alpr_tracker_config.txt"

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define CUSTOM_PTS 1

/* Tracker config parsing */
#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define CONFIG_GROUP_TRACKER_ENABLE_PAST_FRAME "enable-past-frame"
#define CONFIG_GPU_ID "gpu-id"

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }

gint frame_number = 0;

#define SIZE 256

/*config vars*/  
gint muxer_width = 0;
gint muxer_height = 0;
gint muxer_live_source = 0;
gint muxer_batch_size = 0;
gint muxer_batched_push_timeout = 0;
gint muxer_nvbuf_memory_type = 0;
gint open_everyobject_output = 0;
gint open_everycar_classification = 0;
gint lpr_word_limit = 0;
gint lpr_word_count = 0;

/* These are the strings of the labels for the respective models */
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person", "RoadSign" };

gchar sgie2_classes_str[12][32] = { "black", "blue", "brown", "gold", "green",
  "grey", "maroon", "orange", "red", "silver", "white", "yellow"
};

gchar sgie3_classes_str[20][32] = { "Acura", "Audi", "BMW", "Chevrolet", "Chrysler",
  "Dodge", "Ford", "GMC", "Honda", "Hyundai", "Infiniti", "Jeep", "Kia",
      "Lexus", "Mazda", "Mercedes", "Nissan",
  "Subaru", "Toyota", "Volkswagen"
};

gchar sgie4_classes_str[6][32] = { "coupe", "largevehicle", "sedan", "suv",
  "truck", "van"
};

/* Structure to contain all our information for appsrc,
 * so we can pass it to callbacks */
typedef struct _AppSrcData
{
  GstElement *app_source;
  long frame_size;
  FILE *file;                   /* Pointer to the raw video file */
  gint appsrc_frame_num;
  guint fps;                    /* To set the FPS value */
  guint sourceid;               /* To control the GSource */
} AppSrcData;

static void
readConfig(){ 
  char name[SIZE];
  char value[SIZE];
  
  FILE *fp = fopen(CONFIG_PATH, "r");
  if (fp == NULL){
    return;
  }else{
    while(!feof(fp)){
      memset(name,0,SIZE);
      memset(value,0,SIZE);

      /*Read Data*/
      fscanf(fp,"%s = %s\n", name, value);

      if(!strcmp(name, "muxer_width"))
      {
        muxer_width = atoi(value);
      }
      else if(!strcmp(name, "muxer_height"))
      {
        muxer_height = atoi(value);
      }
      else if(!strcmp(name, "muxer_live_source"))
      {
        muxer_live_source = atoi(value);
      }
      else if(!strcmp(name, "muxer_batch_size"))
      {
        muxer_batch_size = atoi(value);
      }
      else if(!strcmp(name, "muxer_batched_push_timeout"))
      {
        muxer_batched_push_timeout = atoi(value);
      }
      else if(!strcmp(name, "muxer_nvbuf_memory_type"))
      {
        muxer_nvbuf_memory_type = atoi(value);
      }
      else if(!strcmp(name, "open_everyobject_output")){
        open_everyobject_output = atoi(value);
      }
      else if(!strcmp(name, "open_everycar_classification")){
        open_everycar_classification = atoi(value);
      }
      else if(!strcmp(name, "lpr_word_limit")){
        lpr_word_limit = atoi(value);
      }
      else if(!strcmp(name, "lpr_word_count")){
        lpr_word_count = atoi(value);
      }
    }
  }
  fclose(fp);
 
  return;
}

/* new_sample is an appsink callback that will extract metadata received
 * tee sink pad and update params for drawing rectangle,
 *object information etc. */
static GstFlowReturn
new_sample (GstElement * sink, gpointer * data)
{
  GstSample *sample;
  GstBuffer *buf = NULL;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  unsigned long int pts = 0;

  sample = gst_app_sink_pull_sample (GST_APP_SINK (sink));
  if (gst_app_sink_is_eos (GST_APP_SINK (sink))) {
    g_print ("EOS received in Appsink********\n");
  }

  if (sample) {
    /* Obtain GstBuffer from sample and then extract metadata from it. */
    buf = gst_sample_get_buffer (sample);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
        l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
      pts = frame_meta->buf_pts;
      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
          l_obj = l_obj->next) {
        obj_meta = (NvDsObjectMeta *) (l_obj->data);
        if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
          vehicle_count++;
          num_rects++;
        }
        if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
          person_count++;
          num_rects++;
        }
      }
    }

    // g_print ("Frame Number = %d Number of objects = %d "
    //     "Vehicle Count = %d Person Count = %d PTS = %" GST_TIME_FORMAT "\n",
    //     frame_number, num_rects, vehicle_count, person_count,
    //     GST_TIME_ARGS (pts));
    frame_number++;
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }
  return GST_FLOW_ERROR;
}

/* This method is called by the idle GSource in the mainloop, 
 * to feed one raw video frame into appsrc.
 * The idle handler is added to the mainloop when appsrc requests us
 * to start sending data (need-data signal)
 * and is removed when appsrc has enough data (enough-data signal).
 */
static gboolean
read_data (AppSrcData * data)
{
  GstBuffer *buffer;
  GstFlowReturn gstret;

  size_t ret = 0;
  GstMapInfo map;
  buffer = gst_buffer_new_allocate (NULL, data->frame_size, NULL);

  gst_buffer_map (buffer, &map, GST_MAP_WRITE);
  ret = fread (map.data, 1, data->frame_size, data->file);
  map.size = ret;

  gst_buffer_unmap (buffer, &map);
  if (ret > 0) {
#if CUSTOM_PTS
    GST_BUFFER_PTS (buffer) =
        gst_util_uint64_scale (data->appsrc_frame_num, GST_SECOND, data->fps);
#endif
    gstret = gst_app_src_push_buffer ((GstAppSrc *) data->app_source, buffer);
    if (gstret != GST_FLOW_OK) {
      g_print ("gst_app_src_push_buffer returned %d \n", gstret);
      return FALSE;
    }
  } else if (ret == 0) {
    gstret = gst_app_src_end_of_stream ((GstAppSrc *) data->app_source);
    if (gstret != GST_FLOW_OK) {
      g_print
          ("gst_app_src_end_of_stream returned %d. EoS not queued successfully.\n",
          gstret);
      return FALSE;
    }
  } else {
    g_print ("\n failed to read from file\n");
    return FALSE;
  }
  data->appsrc_frame_num++;

  return TRUE;
}

/* This signal callback triggers when appsrc needs data. Here,
 * we add an idle handler to the mainloop to start pushing
 * data into the appsrc */
static void
start_feed (GstElement * source, guint size, AppSrcData * data)
{
  if (data->sourceid == 0) {
    data->sourceid = g_idle_add ((GSourceFunc) read_data, data);
  }
}

/* This callback triggers when appsrc has enough data and we can stop sending.
 * We remove the idle handler from the mainloop */
static void
stop_feed (GstElement * source, AppSrcData * data)
{
  if (data->sourceid != 0) {
    g_source_remove (data->sourceid);
    data->sourceid = 0;
  }
}

/* sgie4_src_pad_buffer_probe  will extract metadata received from sgie
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
sgie4_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  guint64 track_id = 0;
  guint64 vehicle_track_id = 0;
  gchar license_plate[SIZE] = {0};
  float lpr_confidence = 0;
  gchar vehicle_color[SIZE] = {0};
  gchar vehicle_make[SIZE] = {0};
  gchar vehicle_type[SIZE] = {0};
  static guint use_device_mem = 0;

  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) 
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    if(frame_meta->obj_meta_list == NULL)
    {
      continue;
    }

    float obj_lpr_box_left = 0;
    float obj_lpr_box_top = 0;
    float obj_lpr_box_height = 0;
    float obj_lpr_box_width = 0;
    float obj_car_box_left = 0;
    float obj_car_box_top = 0;
    float obj_car_box_height = 0;
    float obj_car_box_width = 0;

    for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) 
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) (l_obj->data);
      if(obj_meta->classifier_meta_list == NULL)
      {
        continue;
      }

      vehicle_track_id = 0;
      memset(vehicle_color,0,SIZE);
      memset(vehicle_make,0,SIZE);
      memset(vehicle_type,0,SIZE);

      for (NvDsMetaList * l_class = obj_meta->classifier_meta_list; l_class != NULL; l_class = l_class->next) 
      {
        NvDsClassifierMeta *class_meta = (NvDsClassifierMeta *)(l_class->data);
        if(class_meta->label_info_list == NULL)
        {
          continue;
        }

        for (NvDsMetaList * l_label = class_meta->label_info_list; l_label != NULL; l_label = l_label->next) 
        {
          NvDsLabelInfo *label_info = (NvDsLabelInfo *)l_label->data;

          if(class_meta->unique_component_id == 3)
          {
            if(lpr_word_limit)
            {
              if(strlen(label_info->result_label) == lpr_word_count)
              {
                track_id = obj_meta->parent->object_id;
                strcpy(license_plate, label_info->result_label);
                lpr_confidence = obj_meta->confidence;
                obj_lpr_box_left = obj_meta->detector_bbox_info.org_bbox_coords.left;
                obj_lpr_box_top = obj_meta->detector_bbox_info.org_bbox_coords.top;
                obj_lpr_box_width = obj_meta->detector_bbox_info.org_bbox_coords.width;
                obj_lpr_box_height = obj_meta->detector_bbox_info.org_bbox_coords.height;
              }
            }
            else
            {
              track_id = obj_meta->parent->object_id;
              strcpy(license_plate, label_info->result_label);
              lpr_confidence = obj_meta->confidence;
              obj_lpr_box_left = obj_meta->detector_bbox_info.org_bbox_coords.left;
              obj_lpr_box_top = obj_meta->detector_bbox_info.org_bbox_coords.top;
              obj_lpr_box_width = obj_meta->detector_bbox_info.org_bbox_coords.width;
              obj_lpr_box_height = obj_meta->detector_bbox_info.org_bbox_coords.height;
            }
          }
          else
          {
            vehicle_track_id = obj_meta->object_id;
            if(class_meta->unique_component_id == 4)
            {
              strcpy(vehicle_color, label_info->result_label);
            }
            else if(class_meta->unique_component_id == 5)
            {
              strcpy(vehicle_make, label_info->result_label);
            }
            else if(class_meta->unique_component_id == 6)
            {
              strcpy(vehicle_type, label_info->result_label);
            }
            obj_car_box_left = obj_meta->detector_bbox_info.org_bbox_coords.left;
            obj_car_box_top = obj_meta->detector_bbox_info.org_bbox_coords.top;
            obj_car_box_width = obj_meta->detector_bbox_info.org_bbox_coords.width;
            obj_car_box_height = obj_meta->detector_bbox_info.org_bbox_coords.height;
          }
        }
      }

      float vehicle_color_maxProbability = 0;
      float vehicle_make_maxProbability = 0;
      float vehicle_type_maxProbability = 0;
      bool classification_display_check = false;

      /* Iterate user metadata in object to search SGIE's tensor data */
      for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;  l_user = l_user->next)
      {
        // g_print("Raw meta output of obj_id = %d \n", obj_meta->class_id);
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
          continue;

        /* convert to tensor metadata */
        NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
        for (unsigned int i = 0; i < meta->num_output_layers; i++)
        {
          NvDsInferLayerInfo *info = &meta->output_layers_info[i];
          info->buffer = meta->out_buf_ptrs_host[i];
          if (use_device_mem && meta->out_buf_ptrs_dev[i])
          {
            cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i], 
            info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
          }
        }

        NvDsInferDimsCHW dims;
        getDimsCHWFromDims (dims, meta->output_layers_info[0].inferDims);
        unsigned int numClasses = dims.c;
        float *outputCoverageBuffer = (float *) meta->output_layers_info[0].buffer;

        for (unsigned int c = 0; c < numClasses; c++)
        {
          float probability = outputCoverageBuffer[c];
          if(meta->unique_id == 4)
          {
            classification_display_check = true;
            if (probability > vehicle_color_maxProbability)
            {
              vehicle_color_maxProbability = probability;
              if(strcmp(vehicle_color, "") == 0)
              {
                vehicle_color_maxProbability = 0;
              }
            }
          }
          else if(meta->unique_id == 5)
          {
            if (probability > vehicle_make_maxProbability)
            {
              vehicle_make_maxProbability = probability;
              if(strcmp(vehicle_make, "") == 0)
              {
                vehicle_make_maxProbability = 0;
              }
            }
          }
          else if(meta->unique_id == 6)
          {
            if (probability > vehicle_type_maxProbability)
            {
              vehicle_type_maxProbability = probability;
              if(strcmp(vehicle_type, "") == 0)
              {
                vehicle_type_maxProbability = 0;
              }
            }
          }
        }

        if(!open_everyobject_output)
        {
          if(strcmp(license_plate, "") != 0 && classification_display_check)
          {
            if(track_id == vehicle_track_id)
            {
              g_print("%ld,%s,%f,%s,%f,%s,%f,%s,%f\n", vehicle_track_id, license_plate, lpr_confidence, vehicle_color, vehicle_color_maxProbability, vehicle_make, vehicle_make_maxProbability, vehicle_type, vehicle_type_maxProbability);
              classification_display_check = false;
            }
          }
          // // although not classification, still output car plate
          // else if(strcmp(license_plate, "") != 0 && !classification_display_check)
          // {
          //   g_print("%ld,%s,%f,%s,%d,%s,%d,%s,%d\n", track_id, license_plate, lpr_confidence, "", 0, "", 0, "", 0);
          // }
          else if(strcmp(license_plate, "") == 0 && classification_display_check && open_everycar_classification)
          {
            g_print("%ld,%s,%f,%s,%f,%s,%f,%s,%f\n", vehicle_track_id, license_plate, lpr_confidence, vehicle_color, vehicle_color_maxProbability, vehicle_make, vehicle_make_maxProbability, vehicle_type, vehicle_type_maxProbability);
            classification_display_check = false;
          }
        }
      }

      if(open_everyobject_output)
      {
        vehicle_color_maxProbability = 0;
        vehicle_make_maxProbability = 0;
        vehicle_type_maxProbability = 0;

        if(strcmp(license_plate, "") != 0)
        {
          if(track_id == vehicle_track_id)
          {
            g_print("%ld,%s,%f,%s,%f,%s,%f,%s,%f\n", vehicle_track_id, license_plate, lpr_confidence, vehicle_color, vehicle_color_maxProbability, vehicle_make, vehicle_make_maxProbability, vehicle_type, vehicle_type_maxProbability);
          }
        }
        else if(strcmp(license_plate, "") == 0 && open_everycar_classification)
        {
          g_print("%ld,%s,%f,%s,%f,%s,%f,%s,%f\n", vehicle_track_id, license_plate, lpr_confidence, vehicle_color, vehicle_color_maxProbability, vehicle_make, vehicle_make_maxProbability, vehicle_type, vehicle_type_maxProbability);
        }
      }

    }
  }

  use_device_mem = 1 - use_device_mem;
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static gchar *
get_absolute_file_path (gchar *cfg_file_path, gchar *file_path)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  if (file_path && file_path[0] == '/') {
    return file_path;
  }

  if (!realpath (cfg_file_path, abs_cfg_path)) {
    g_free (file_path);
    return NULL;
  }

  // Return absolute path of config file if file_path is NULL.
  if (!file_path) {
    abs_file_path = g_strdup (abs_cfg_path);
    return abs_file_path;
  }

  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  abs_file_path = g_strconcat (abs_cfg_path, file_path, NULL);
  g_free (file_path);

  return abs_file_path;
}

static gboolean
set_tracker_properties (GstElement *nvtracker)
{
  gboolean ret = FALSE;
  GError *error = NULL;
  gchar **keys = NULL;
  gchar **key = NULL;
  GKeyFile *key_file = g_key_file_new ();

  if (!g_key_file_load_from_file (key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE,
          &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    return FALSE;
  }

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TRACKER, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_WIDTH)) {
      gint width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_WIDTH, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-width", width, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_HEIGHT)) {
      gint height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_HEIGHT, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-height", height, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      guint gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "gpu_id", gpu_id, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE)) {
      char* ll_config_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-config-file", ll_config_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
      char* ll_lib_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-lib-file", ll_lib_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS)) {
      gboolean enable_batch_process =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "enable-batch-process",
                    enable_batch_process, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_ENABLE_PAST_FRAME)) {
      gboolean enable_past_frame =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_ENABLE_PAST_FRAME, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "enable-past-frame",
                    enable_past_frame, NULL);
    } else {
      g_printerr ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TRACKER);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    g_printerr ("%s failed", __func__);
  }
  return ret;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *nvvidconv1 = NULL, *caps_filter = NULL,
      *streammux = NULL, *sink = NULL, *pgie = NULL, *nvtracker = NULL, 
      *sgie0 = NULL, *sgie1 = NULL, *sgie2 = NULL, *sgie3 = NULL, *sgie4 = NULL, *nvvidconv2 = NULL,
      *nvosd = NULL, *tee = NULL, *appsink = NULL;
  GstElement *transform = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  AppSrcData data;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;
  gchar *endptr1 = NULL, *vidconv_format = NULL;
  GstPad *tee_source_pad1, *tee_source_pad2;
  GstPad *osd_sink_pad, *appsink_sink_pad;

  readConfig();

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);
  /* Check input arguments */
  if (argc != 4) {
    g_printerr
        ("Usage: %s <Raw filename> <fps> <format(I420, NV12, RGBA)>\n",
        argv[0]);
    return -1;
  }

  long fps = g_ascii_strtoll (argv[2], &endptr1, 10);
  gchar *format = argv[3];
  if (fps == 0 && endptr1 == argv[2]) {
    g_printerr ("Incorrect FPS\n");
    return -1;
  }

  if (fps == 0) {
    g_printerr ("FPS cannot be 0\n");
    return -1;
  }

  if (g_strcmp0 (format, "I420") != 0 && g_strcmp0 (format, "RGBA") != 0
      && g_strcmp0 (format, "NV12") != 0) {
    g_printerr ("Only I420, RGBA and NV12 are supported\n");
    return -1;
  }

  /* Initialize custom data structure */
  memset (&data, 0, sizeof (data));
  if (!g_strcmp0 (format, "RGBA")) {
    data.frame_size = muxer_width * muxer_height * 4;
    vidconv_format = "RGBA";
  } else {
    data.frame_size = muxer_width * muxer_height * 1.5;
    vidconv_format = "NV12";
  }
  data.file = fopen (argv[1], "r");
  data.fps = fps;

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("alpr-appsrc-pipeline");
  if (!pipeline) {
    g_printerr ("Pipeline could not be created. Exiting.\n");
    return -1;
  }

  /* App Source element for reading from raw video file */
  data.app_source = gst_element_factory_make ("appsrc", "app-source");
  if (!data.app_source) {
    g_printerr ("Appsrc element could not be created. Exiting.\n");
    return -1;
  }

  /* Use convertor to convert from software buffer to GPU buffer */
  nvvidconv1 =
      gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");
  if (!nvvidconv1) {
    g_printerr ("nvvideoconvert1 could not be created. Exiting.\n");
    return -1;
  }
  caps_filter = gst_element_factory_make ("capsfilter", "capsfilter");
  if (!caps_filter) {
    g_printerr ("Caps_filter could not be created. Exiting.\n");
    return -1;
  }

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
  if (!streammux) {
    g_printerr ("nvstreammux could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on streammux's output,
   * behaviour of inferencing is set through config file */
  /* Create three nvinfer instances for two detectors and one classifier*/
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  if (!pgie) {
    g_printerr ("Primary nvinfer could not be created. Exiting.\n");
    return -1;
  }

  sgie0 = gst_element_factory_make ("nvinfer", "secondary0-nvinference-engine");
  if (!sgie0) {
    g_printerr ("Sgie0 nvinfer could not be created. Exiting.\n");
    return -1;
  }                 

  sgie1 = gst_element_factory_make ("nvinfer", "secondary1-nvinference-engine");
  if (!sgie1) {
    g_printerr ("Sgie1 nvinfer could not be created. Exiting.\n");
    return -1;
  }   

  /* We need to have a tracker to track the identified objects */
  nvtracker = gst_element_factory_make ("nvtracker", "tracker");
  if (!nvtracker) {
    g_printerr ("Tracker nvtracker could not be created. Exiting.\n");
    return -1;
  }

  /* We need three secondary gies so lets create 3 more instances of
     nvinfer */
  sgie2 = gst_element_factory_make ("nvinfer", "secondary2-nvinference-engine");
  if (!sgie2) {
    g_printerr ("Secondary2 nvinfer could not be created. Exiting.\n");
    return -1;
  }

  sgie3 = gst_element_factory_make ("nvinfer", "secondary3-nvinference-engine");
  if (!sgie3) {
    g_printerr ("Secondary3 nvinfer could not be created. Exiting.\n");
    return -1;
  }

  sgie4 = gst_element_factory_make ("nvinfer", "secondary4-nvinference-engine");
  if (!sgie4) {
    g_printerr ("Secondary4 nvinfer could not be created. Exiting.\n");
    return -1;
  }

  /* Use convertor to convert from NV12 to RGBA as required by nvdsosd */
  nvvidconv2 =
      gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");
  if (!nvvidconv2) {
    g_printerr ("nvvideoconvert2 could not be created. Exiting.\n");
    return -1;
  }

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  if (!nvosd) {
    g_printerr ("nvdsosd could not be created. Exiting.\n");
    return -1;
  }

  /* Finally render the osd output. We will use a tee to render video
   * playback on nveglglessink, and we use appsink to extract metadata
   * from buffer and print object, person and vehicle count. */
  tee = gst_element_factory_make ("tee", "tee");
  if (!tee) {
    g_printerr ("Tee could not be created. Exiting.\n");
    return -1;
  }
  if(prop.integrated) {
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
    if (!transform) {
      g_printerr ("Tegra transform element could not be created. Exiting.\n");
      return -1;
    }
  }
  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
  if (!sink) {
    g_printerr ("Display sink could not be created. Exiting.\n");
    return -1;
  }

  appsink = gst_element_factory_make ("appsink", "app-sink");
  if (!appsink) {
    g_printerr ("Appsink element could not be created. Exiting.\n");
    return -1;
  }

  /* Configure appsrc */
  g_object_set (data.app_source, "caps",
      gst_caps_new_simple ("video/x-raw",
          "format", G_TYPE_STRING, format,
          "width", G_TYPE_INT, muxer_width,
          "height", G_TYPE_INT, muxer_height,
          "framerate", GST_TYPE_FRACTION, data.fps, 1, NULL), NULL);
#if !CUSTOM_PTS
  g_object_set (G_OBJECT (data.app_source), "do-timestamp", TRUE, NULL);
#endif
  g_signal_connect (data.app_source, "need-data", G_CALLBACK (start_feed),
      &data);
  g_signal_connect (data.app_source, "enough-data", G_CALLBACK (stop_feed),
      &data);

  caps =
      gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
      vidconv_format, NULL);
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);
  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);

  /* Set streammux properties */
  g_object_set (G_OBJECT (streammux), "width", muxer_width, "height",
      muxer_height, "batch-size", muxer_batch_size, "live-source", muxer_live_source,
      "batched-push-timeout", muxer_batched_push_timeout, "nvbuf-memory-type", muxer_nvbuf_memory_type, NULL);

  /* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
  g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT (sgie0), "config-file-path", SGIE0_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT (sgie1), "config-file-path", SGIE1_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT (sgie2), "config-file-path", SGIE2_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT (sgie3), "config-file-path", SGIE3_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT (sgie4), "config-file-path", SGIE4_CONFIG_FILE, NULL);

  /* Set necessary properties of the tracker element. */
  if (!set_tracker_properties(nvtracker)) {
    g_printerr ("Failed to set tracker properties. Exiting.\n");
    return -1;
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline),
      data.app_source, nvvidconv1, caps_filter, streammux, pgie, nvtracker, sgie0, sgie1, sgie2, sgie3, sgie4,
      nvvidconv2, nvosd, tee, sink, appsink, NULL);
  if(prop.integrated) {
    gst_bin_add (GST_BIN (pipeline), transform);
  }

  GstPad *src_pad6;
  src_pad6 = gst_element_get_static_pad (sgie4, "src");
  if (!src_pad6)
    g_print ("Unable to get secondary_gie4 src pad\n");
  else
  {
    gst_pad_add_probe(src_pad6, GST_PAD_PROBE_TYPE_BUFFER, sgie4_src_pad_buffer_probe, NULL, NULL);
    gst_object_unref (src_pad6);
  }

  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad (caps_filter, pad_name_src);
  if (!srcpad) {
    g_printerr ("Caps filter request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link caps filter to stream muxer. Exiting.\n");
    return -1;
  }

  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  /* we link the elements together */
  /* app-source -> nvvidconv -> caps filter ->
   * nvinfer -> nvvidconv -> nvosd -> video-renderer */
  if(prop.integrated) {
    if (!gst_element_link_many (data.app_source, nvvidconv1, caps_filter, NULL) ||
        !gst_element_link_many (nvosd, transform, sink, NULL) ||
        !gst_element_link_many (streammux, pgie, nvtracker, sgie0, sgie1, sgie2, sgie3, sgie4, nvvidconv2, tee, NULL)) {
      g_printerr ("Elements could not be linked: Exiting.\n");
      return -1;
    }
  }
  else {
    if (!gst_element_link_many (data.app_source, nvvidconv1, caps_filter, NULL) ||
        !gst_element_link_many (nvosd, sink, NULL) ||
        !gst_element_link_many (streammux, pgie, nvtracker, sgie0, sgie1, sgie2, sgie3, sgie4, nvvidconv2, tee, NULL)) {
      g_printerr ("Elements could not be linked: Exiting.\n");
      return -1;
    }
  }

/* Manually link the Tee, which has "Request" pads.
 * This tee, in case of multistream usecase, will come before tiler element. */
  tee_source_pad1 = gst_element_get_request_pad (tee, "src_0");
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  tee_source_pad2 = gst_element_get_request_pad (tee, "src_1");
  appsink_sink_pad = gst_element_get_static_pad (appsink, "sink");
  if (gst_pad_link (tee_source_pad1, osd_sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Tee could not be linked to display sink.\n");
    gst_object_unref (pipeline);
    return -1;
  }
  if (gst_pad_link (tee_source_pad2, appsink_sink_pad) != GST_PAD_LINK_OK) {
    g_printerr ("Tee could not be linked to appsink.\n");
    gst_object_unref (pipeline);
    return -1;
  }
  
  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */  
  if (!osd_sink_pad)
    g_print ("Unable to get osd sink pad\n");
  gst_object_unref (osd_sink_pad);

  if (!appsink_sink_pad) {
    g_printerr ("Unable to get appsink sink pad\n");
  }
  gst_object_unref (appsink_sink_pad);

  /* Configure appsink to extract data from DeepStream pipeline */
  g_object_set (appsink, "emit-signals", TRUE, "async", FALSE, NULL);
  g_object_set (sink, "sync", FALSE, NULL);

  /* Callback to access buffer and object info. */
  g_signal_connect (appsink, "new-sample", G_CALLBACK (new_sample), NULL);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing: %s\n", argv[1]);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
