import tensorflow as tf
from keras.src.datasets import cifar10
import numpy as np
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit 
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

engine_file = "model_color.trt"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_file):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file, "rb") as f:
        engine_data = f.read()

    engine = runtime.deserialize_cuda_engine(engine_data) # loading engine in vram
    return engine

engine = load_engine(engine_file)

context = engine.create_execution_context()

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

num_classes = tf.reduce_max(tf.cast(train_labels, tf.int32)) + 1
train_labels_1h = tf.one_hot( tf.cast(train_labels, tf.int32)[:, 0], depth=num_classes)
test_labels_1h = tf.one_hot( tf.cast(test_labels, tf.int32)[:, 0], depth=num_classes)

test_images_batched = tf.expand_dims(test_images, axis=1).numpy().astype(np.float32)

test_labels_1h_batched = tf.expand_dims(test_labels_1h, axis=1).numpy().astype(np.float32)

# input_memory_total_shape = test_images_batched.shape # (10000, 1, 32, 32, 3)
d_input = cuda.mem_alloc(test_images_batched.nbytes)

# output_memory_total_shape = test_labels_1h_batched.shape # (10000, 1, 10)
d_output = cuda.mem_alloc(test_labels_1h_batched.nbytes)

# input_memory_individual_shape = test_images_batched[0].shape # (1, 32, 32, 3)
input_skip_bytes = test_images_batched[0].nbytes

# output_memory_individual_shape = (1, 10)
output_skip_bytes = test_labels_1h_batched[0].nbytes

cuda.memcpy_htod(d_input, test_images_batched)

def benchmark_inference_time(ctx, num_runs=100):
    
    start_time = time.time()
    for i in range(num_runs):
        ctx.execute_v2([int(d_input) + i * input_skip_bytes, int(d_output) + i * output_skip_bytes])
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_runs
    return avg_inference_time

def benchmark_thoughput(ctx, total_batches = 10000):
    
    start_time = time.time()
    for i in range(total_batches):
        ctx.execute_v2([int(d_input) + i * input_skip_bytes, int(d_output) + i * output_skip_bytes])
    end_time = time.time()

    start_time_overhead = time.time()
    for i in range(total_batches):
        pass
    end_time_overhead = time.time()

    print("overhead (for loop on python): ", end_time_overhead - start_time_overhead)
    print("time: ", end_time - start_time)

    total_time = (end_time - start_time) - (end_time_overhead - start_time_overhead)

    thoughput = total_batches / total_time

    return thoughput

def benchmark_accuracy_score(ctx):
    
    for i in range(test_images_batched.shape[0]):
        ctx.execute_v2([int(d_input) + i * input_skip_bytes, int(d_output) + i * output_skip_bytes])
    
    output_data = np.empty(test_labels_1h_batched.shape, dtype=np.float32)
    cuda.memcpy_dtoh(output_data, d_output)

    output_data = np.argmax(output_data, axis=2)
    test_labels_1h = np.argmax(test_labels_1h_batched, axis=2)

    acc = accuracy_score(test_labels_1h, output_data)
    prec = precision_score(test_labels_1h, output_data, average='macro', zero_division=0)
    rec = recall_score(test_labels_1h, output_data, average='macro', zero_division=0)
    f1 = f1_score(test_labels_1h, output_data, average='macro', zero_division=0)

    print(f"Accuracy: {acc:.6f}")
    print(f"Macro-Average Precision: {prec:.6f}")
    print(f"Macro-Average Recall: {rec:.6f}")
    print(f"Macro-Average F1: {f1:.6f}")

    print("classification report:\n", classification_report(test_labels_1h, output_data, zero_division=0))

    print("confusion matrix:\n", confusion_matrix(test_labels_1h, output_data))


if __name__ == "__main__":
    
    try:
        # avg_inference_time = benchmark_inference_time(context, 10000)
        # print("Average inference time (in ms): ", avg_inference_time * 1000)

        # thoughput = benchmark_thoughput(context, test_images_batched.shape[0])
        # print("Thoughput: ", thoughput, " images/second")

        benchmark_accuracy_score(context)

    except Exception as e:
        print(e)
