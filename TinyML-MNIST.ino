#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

#include "mnist_cnn.h"
#include "x_test.h"

#define N_INPUTS sizeof(x_test_dat)
#define N_OUTPUTS 10
// preallocate a certain amount of memory for input, output, and intermediate arrays.
#define TENSOR_ARENA_SIZE 64*1024 

Eloquent::TinyML::TensorFlow::TensorFlow<N_INPUTS, N_OUTPUTS, TENSOR_ARENA_SIZE> tf;

float x_test[N_INPUTS];

void setup() {
  Serial.begin(115200);
  tf.begin(mnist_cnn);

  for (int i = 0; i < N_INPUTS; i++) {
    x_test[i] = x_test_dat[i] / 255.0; // normalize x_test
  }
}

void loop() {
  float y_pred[10] ={ 0 };

  uint32_t start = micros(); 

  tf.predict(x_test, y_pred);

  uint32_t timeit = micros() - start;

  Serial.print("It took ");
  Serial.print(timeit);
  Serial.println(" micros to run inference");

  for (int i = 0; i < N_OUTPUTS; i++) {
    Serial.print(y_pred[i]);
    Serial.print(i == N_OUTPUTS - 1 ? '\n' : ',');
  }

  Serial.print("Predicted Class :");
  Serial.println(tf.probaToClass(y_pred));
  Serial.print("Sanity check:");
  Serial.println(tf.predictClass(x_test));

  delay(5000);
}
