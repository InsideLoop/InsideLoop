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

#ifndef IL_QUEUE_H
#define IL_QUEUE_H

#include <il/Array.h>

namespace il {

template <typename T>
class Queue {
 private:
  il::Array<T> data_;
  // Index of the next element to be served
  il::int_t front_;
  // Index of the last element in the queue
  il::int_t back_;

 public:
  Queue() : data_{} {
    front_ = 0;
    back_ = -1;
  }
  Queue(il::int_t n) : data_{n} {
    front_ = 0;
    back_ = -1;
  }
  il::int_t size() const {
    if (front_ == 0 && back_ == 1) {
      return 0;
    } else if (front_ <= back_) {
      return back_ - front_ + 1;
    } else {
      return back_ + 1 + data_.size() - front_;
    }
  }
  void Reserve(il::int_t n) {
    IL_ASSERT(data_.size() == 0 && front_ == 0 && back_ == -1);

    data_.Resize(n);
    front_ = 0;
    back_ = -1;
  }
  il::int_t capacity() const {
    return data_.size();
  }
  void Append(const T& x) {
    if (back_ == -1 && front_ == 0) {
      data_[0] = x;
      front_ = 0;
      back_ = 0;
    } else if (back_ + 1 < front_ || back_ + 1 < data_.size()) {
      ++back_;
      data_[back_] = x;
    } else if (back_ + 1 == data_.size()) {
      if (front_ > 0) {
        data_[0] = x;
        back_ = 0;
      } else {
        IL_UNREACHABLE;
      }
    }
  }
  template <typename... Args>
  void Append(il::emplace_t, Args&&... args) {
    if (back_ == -1 && front_ == 0) {
      data_[0] = T(std::forward<Args>(args)...);
      front_ = 0;
      back_ = 0;
    } else if (back_ + 1 < front_ || back_ + 1 < data_.size()) {
      ++back_;
      data_[back_] = T(std::forward<Args>(args)...);
    } else if (back_ + 1 == data_.size()) {
      if (front_ > 0) {
        data_[0] = T(std::forward<Args>(args)...);
        back_ = 0;
      } else {
        IL_UNREACHABLE;
      }
    }
  }
  const T& front() const {
    return data_[front_];
  }
  T Pop() {
    if (front_ + 1 < data_.size()) {
      ++front_;
      return data_[front_ - 1];
    } else {
      front_ = 0;
      return data_[data_.size() - 1];
    }
  };
};


}  // namespace il

#endif  // IL_QUEUE_H
