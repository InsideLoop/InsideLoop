//==============================================================================
//
// Copyright 2017 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//==============================================================================

#ifndef IL_PPM_H
#define IL_PPM_H

#include <cstdio>
#include <string>

#include <il/Array2D.h>
#include <il/StaticArray.h>
#include <il/Status.h>

namespace il {

struct Pixel {
  unsigned char red;
  unsigned char green;
  unsigned char blue;
};

il::Array2D<il::Pixel> readPpm(const std::string& filename, il::io_t,
                               il::Status& status) {
  il::Array2D<il::Pixel> image{};

#ifdef IL_UNIX
  FILE* fp = std::fopen(filename.c_str(), "rb");
  if (!fp) {
    status.setError(il::Error::FilesystemFileNotFound);
    return image;
  }
#else
  FILE* fp;
  const errno_t error_nb = fopen_s(&fp, filename.c_str(), "rb");
  if (error_nb != 0) {
    status.setError(il::Error::FilesystemFileNotFound);
    return image;
  }
#endif

  char buffer[16];
  if (!std::fgets(buffer, sizeof(buffer), fp)) {
    status.set(ErrorCode::BinaryFileWrongFormat);
    return image;
  }
  if (buffer[0] != 'P' || buffer[1] != '6') {
    status.set(ErrorCode::BinaryFileWrongFormat);
    return image;
  }

  // Check for comments
  int c{std::getc(fp)};
  while (c == '#') {
    while (std::getc(fp) != '\n') {
    };
    c = std::getc(fp);
  }
  std::ungetc(c, fp);

  // read image size information
  int width;
  int height;
#ifdef IL_UNIX
  if (std::fscanf(fp, "%d %d", &width, &height) != 2) {
    status.setError(il::Error::BinaryFileWrongFormat);
    return image;
  }
#else
  const int error_no = fscanf_s(fp, "%d %d", &width, &height);
  if (error_no != 2) {
    status.setError(il::Error::BinaryFileWrongFormat);
    return image;
  }
#endif
  // read rgb component
  int rgb_comp_color;
#ifdef IL_UNIX
  if (std::fscanf(fp, "%d", &rgb_comp_color) != 1) {
    status.setError(il::Error::BinaryFileWrongFormat);
    return image;
  }
#else
  {
    const int error_no = fscanf_s(fp, "%d", &rgb_comp_color);
    if (error_no != 1) {
      status.setError(il::Error::BinaryFileWrongFormat);
      return image;
    }
  }
#endif
  // check rgb component depth
  if (rgb_comp_color != 255) {
    status.setError(il::Error::BinaryFileWrongFormat);
    return image;
  }
  while (std::fgetc(fp) != '\n') {
  };

  // read pixel data from file
  image.resize(width, height);
  if (std::fread(image.data(), 3 * width, height, fp) != height) {
    status.setError(il::Error::BinaryFileWrongFormat);
    image.resize(0, 0);
    return image;
  }

  std::fclose(fp);
  status.set(ErrorCode::ok);

  return image;
}
}  // namespace il

#endif  // IL_PPM_H

// From Stackoverflow
// read PPM file and store it in an array; coded with C
// http://stackoverflow.com/questions/2693631/read-ppm-file-and-store-it-in-an-array-coded-with-c
//
//#include<stdio.h>
//#include<stdlib.h>
//
// typedef struct {
//  unsigned char red,green,blue;
//} PPMPixel;
//
// typedef struct {
//  int x, y;
//  PPMPixel *data;
//} PPMImage;
//
//#define CREATOR "RPFELGUEIRAS"
//#define RGB_COMPONENT_COLOR 255
//
// static PPMImage *readPPM(const char *filename)
//{
//  char buff[16];
//  PPMImage *img;
//  FILE *fp;
//  int c, rgb_comp_color;
//  //open PPM file for reading
//  fp = fopen(filename, "rb");
//  if (!fp) {
//    fprintf(stderr, "Unable to open file '%s'\n", filename);
//    exit(1);
//  }
//
//  //read image format
//  if (!fgets(buff, sizeof(buff), fp)) {
//    perror(filename);
//    exit(1);
//  }
//
//  //check the image format
//  if (buff[0] != 'P' || buff[1] != '6') {
//    fprintf(stderr, "Invalid image format (must be 'P6')\n");
//    exit(1);
//  }
//
//  //alloc memory form image
//  img = (PPMImage *)malloc(sizeof(PPMImage));
//  if (!img) {
//    fprintf(stderr, "Unable to allocate memory\n");
//    exit(1);
//  }
//
//  //check for comments
//  c = getc(fp);
//  while (c == '#') {
//    while (getc(fp) != '\n') ;
//    c = getc(fp);
//  }
//
//  ungetc(c, fp);
//  //read image size information
//  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
//    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
//    exit(1);
//  }
//
//  //read rgb component
//  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
//    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
//    exit(1);
//  }
//
//  //check rgb component depth
//  if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
//    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
//    exit(1);
//  }
//
//  while (fgetc(fp) != '\n') ;
//  //memory allocation for pixel data
//  img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));
//
//  if (!img) {
//    fprintf(stderr, "Unable to allocate memory\n");
//    exit(1);
//  }
//
//  //read pixel data from file
//  if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
//    fprintf(stderr, "Error loading image '%s'\n", filename);
//    exit(1);
//  }
//
//  fclose(fp);
//  return img;
//}
// void writePPM(const char *filename, PPMImage *img)
//{
//  FILE *fp;
//  //open file for output
//  fp = fopen(filename, "wb");
//  if (!fp) {
//    fprintf(stderr, "Unable to open file '%s'\n", filename);
//    exit(1);
//  }
//
//  //write the header file
//  //image format
//  fprintf(fp, "P6\n");
//
//  //comments
//  fprintf(fp, "# Created by %s\n",CREATOR);
//
//  //image size
//  fprintf(fp, "%d %d\n",img->x,img->y);
//
//  // rgb component depth
//  fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);
//
//  // pixel data
//  fwrite(img->data, 3 * img->x, img->y, fp);
//  fclose(fp);
//}
//
// void changeColorPPM(PPMImage *img)
//{
//  int i;
//  if(img){
//
//    for(i=0;i<img->x*img->y;i++){
//      img->data[i].red=RGB_COMPONENT_COLOR-img->data[i].red;
//      img->data[i].green=RGB_COMPONENT_COLOR-img->data[i].green;
//      img->data[i].blue=RGB_COMPONENT_COLOR-img->data[i].blue;
//    }
//  }
//}
//
// int main(){
//  PPMImage *image;
//  image = readPPM("can_bottom.ppm");
//  changeColorPPM(image);
//  writePPM("can_bottom2.ppm",image);
//  printf("Press any key...");
//  getchar();
//}
