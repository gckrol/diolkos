# Benchmarks

* GMAC: giga mulitiply-accumulates per second
* TOPS: tera-operations per second. 1 TOPS is 1000 GMAC
* tok/s: tokens/second

Unless specified otherwise, benchmarks were executed 3 times, and the highest value was taken.

Inference benchmarks were done with a Llama2 7B model, quantized to 8 bits, with a BF16 KV cache. They were done with a short prompt ("It was a dark and stormy night"), and with a max of 100 generated tokens.

## Single node

Machine  | GMAC | tok/s | llama.cpp tok/s | Efficiency compared to llama.cpp
---------|------|-------|-----------------|---------------------------------
Laptop 1 | 24.3 | 3.1   | 3.6             | 86%
Laptop 2 | 24.1 |     |
RV2      |  5.6 | 0.7 |

## Cluster

Machines|Total GMAC|Theoretical tok/s|Measured tok/s|Speedup|Efficiency|Notes
-|-|-|-|-|-|-
Laptop 1 100%|24.3|3.1|2.7|-13%|89%|Baseline measurement for overhead
Laptop 1 50% + Laptop 1 50%|24.3|3.1|2.9|-6%|92%|Two workers on the local machine
Laptop 1 50% + Laptop 2 50%|48.4|6.17|3.84|23%|62%|Test showing actual speedup

Workers
    
    bin/worker 1234

Inference started with 

    OMP_NUM_THREADS=<non HT cores> bin/client models/Llama-2-7b-chat-hf -n 100 -i "It was a dark and stormy night

System set to use the performance scheduler using

````
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
performance
````

### Laptop 1 + Laptop 2

````
$ OMP_NUM_THREADS=4 bin/client models/Llama-2-7b-chat-hf -n 100 -i "It was a dark and stormy night"
Processing safetensors file: models/Llama-2-7b-chat-hf/model-00002-of-00002.safetensors
Processing safetensors file: models/Llama-2-7b-chat-hf/model-00001-of-00002.safetensors
Processing safetensors file: models/Llama-2-7b-chat-hf/model.safetensors.index.json
Loaded 291 tensors from safetensors files
Transformer config from safetensors: dim=4096, hidden_dim=11008, n_layers=32, n_heads=32, n_kv_heads=32, vocab_size=32000, seq_len=4096
Total number of matmul_remote calls: 224
Connecting to worker 0 at 127.0.0.1:1234
Connecting to worker 1 at 192.168.178.36:1234
Initialization took 1431 ms
Worker 127.0.0.1:1234: average round-trip latency: 0.019 ms (18.658 us)
Worker 127.0.0.1:1234: network overhead: 0.024 ms (24.165 us) input size: 4608 bytes, output size: 6192 bytes)
Worker 127.0.0.1:1234: network overhead: 1.170 ms (1170.211 us) input size: 4608 bytes, output size: 6192 bytes)
Worker 192.168.178.36:1234: average round-trip latency: 0.164 ms (163.663 us)
Worker 192.168.178.36:1234: network overhead: 0.511 ms (511.345 us) input size: 4608 bytes, output size: 6192 bytes)
Worker 192.168.178.36:1234: network overhead: 1.594 ms (1593.784 us) input size: 4608 bytes, output size: 6192 bytes)

Memory by layer type:
Layer Type           Quant      Memory          Params (M)      Bytes/Param    
Token embedding      Q8_0       140.62 MB       131.07          1.12           
  RMS Attention      F32        0.50 MB         0.13            4.00           
  Query weights      Q8_0       576.00 MB       536.87          1.12           
  Key weights        Q8_0       576.00 MB       536.87          1.12           
  Value weights      Q8_0       576.00 MB       536.87          1.12           
  Output weights     Q8_0       576.00 MB       536.87          1.12           
  RMS FFN            F32        0.50 MB         0.13            4.00           
  W1 weights         Q8_0       1.51 GB         1442.84         1.12           
  W2 weights         Q8_0       1.51 GB         1442.84         1.12           
  W3 weights         Q8_0       1.51 GB         1442.84         1.12           
  Final RMS norm     F32        0.02 MB         0.00            4.00           
Classifier           Q8_0       140.62 MB       131.07          1.12           
Key cache            BF16       1.00 GB         536.87          2.00           
Value cache          BF16       1.00 GB         536.87          2.00           
Total parameters: 6738.42 M, total layers: 32
Total memory: 9.06 GB (runstate: 0.77 MB, kv: 2.00 GB, model: 7.06 GB)

It was a dark and stormy night. The kind of night that made you want to stay inside and huddle under a blanket with a good book. But I had a job to do, so I put on my raincoat and my galoshes and set out into the storm.

As I walked down the street, the wind howled and the rain pounded against my face. I could barely see a few feet in front of me, it
Speed: 3.93 tok/s,  254.323 ms/token 1.135 ms/rpc
GMAC: 26.5
Performance log written to matmul_perf.csv
Worker 127.0.0.1:1234: average round-trip latency: 0.023 ms (22.769 us)
Worker 127.0.0.1:1234: network overhead: 0.026 ms (26.126 us) input size: 4608 bytes, output size: 6192 bytes)
Worker 127.0.0.1:1234: network overhead: 1.232 ms (1231.663 us) input size: 4608 bytes, output size: 6192 bytes)
Worker 192.168.178.36:1234: average round-trip latency: 0.321 ms (320.759 us)
Worker 192.168.178.36:1234: network overhead: 0.496 ms (495.679 us) input size: 4608 bytes, output size: 6192 bytes)
Worker 192.168.178.36:1234: network overhead: 1.573 ms (1573.101 us) input size: 4608 bytes, output size: 6192 bytes)
````

## Test machines

Name     | CPU                          | Memory | Memory bandwidth | Description
---------|------------------------------|--------|------------------|-----------------------------
Laptop 1 | i7-7820HQ CPU @ 2.90GHz      | 32 GB  | 38.4 GB/s        | Old high end Dell laptop
Laptop 2 | i3-1115G4 @ 3.00GHz         | 8 GB   | 51.2 GB/s        | Newer lower end laptop
RV2      | Ky X1 8-core RISC-V @1.6 GHz | 8 GB   | 10.6 GB/s        | OrangePi RV2 RISC-V devboard

Note that the memory bandwidth is the theoretical maximum. Real world results will be 10-20% lower.

## Gathering information

For the exact cpu type, use

    lscpu | grep "Model name"

To find the memory speed, use

    sudo lshw -short -C memory

Then ask AI to compute the bandwidth for you. Verify by checking the max memory bandwidth of the processor.
