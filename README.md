# Diolkos

What if that pile of electronics junk you have could think? Now it can! Diolkos turns your single-board computers, old laptops, washing machines (well - not yet) into a cluster that can run inference at acceptable speeds.

- Are you preparing for after the apocalypse, where supply of electronics is limited and you have to use what you've got?
- Do you want to play around with electronics and inference, and learn how it works?
- Do you want to reduce your dependence on the big players (Nvidia, AMD, Intel, Apple), and run inference on RISC-V or ARM?

Diolkos is a local distributed inference runtime, targeting old or low-powered hardware, including RISC-V/ARM devboards. This reduces dependence on both the cloud and on large non-EU CPU/GPU vendors. The target user is anyone who is able to install Linux. The code aims to be clean and well-documented, so others can build on top of it.

Diolkos started as a fork of [llama2.c](https://github.com/karpathy/llama2.c). Highly recommended if you want to hack on some LLM inference code!

## Roadmap

Project goals:
- High-quality, plain C code
- Runs on all Linux devices, especially tiny ones (RISC-V, ARM boards)
- Uses the CPU, not the GPU, no accelerators
- Good performance
- Small, predictable RAM usage (no crashes while loading)
- Starts up quickly
- Easy to set up and use
- Production-ready (within the constraints of the hardware)
- Easy to hack on

Non-goals (for now):
- GPU or other accelerator support - unless it's 2 lines of code
- Windows/MacOS support

### Phase 1 (completed)

Phase 1 of this project has been completed, and Diolkos currently supports heterogeneous local inference with tensor parallelism, using a star topology. This achieves a small speedup over running inference on a single node. Single-node CPU performance is 90% of llama.cpp, all while using a clean, self-contained C implementation without the use of intrinsics or assembly code.

The code is still highly experimental, but the following features are all working:

- Loading of safetensor models (LLaMA architecture).
- Automatic quantization to Q8_0.
- Efficient caching/loading of quantized models.
- 8-bit inference at 90% of the speed of Llama.cpp.
- Clustered operation at 90% efficiency.

Supported model types:
- LLaMA 2

Supported model formats:
- Non-quantized (F16/BF16/BF32) safetensor. These are widely available on HuggingFace.

### Phase 2

* Move from a star topology to a full (logical) mesh.
* Pipeline all computations.
* Phase out the use of OpenMP for threading, use pthreads directly everywhere. This will reduce overhead and increase performance. #pragma openmp simd will still be used.

### Future phases

Future phases may include:

* Loading GGUF models.
* Loading the tokenizer from the Huggingface model.
* Basic GPU/accelerator support, perhaps from other projects.
* Building & tuning for a benchtop RISC-V cluster.
* Code cleanup & adding comments.

## Comparisons

| Feature/Property                | Petals            | Distributed Llama | Exo              | Llama.cpp        | llama2.cpp       | Diolkos          |
|---------------------------------|-------------------|-------------------|------------------|------------------|------------------|------------------|
| Distributed Computation Type    | Star              | Star              | ?                | None             | None             | Star/Mesh*       |
| Parallelism Type                | Layer             | Tensor            | Layer            | None             | None             | Tensor           |
| Focus on GPU                    | Yes               | Yes               | Yes              | Yes              | No               | No               |
| Focus on CPU                    | No                | Yes               | No               | Yes              | Yes              | Yes              |
| Programming Language            | Python            | C++               | Python           | C++              | C                | C                |

Currently Diolkos uses a star topology, but in the next phase this will change to a mesh topology.

In summary, the goals of this project align most closely with Distributed Llama. This project was evaluated as a base, but the code was deemed too complicated to efficiently experiment with, so instead llama2.c was chosen. Using plain C also makes it easier to control and evaluate generated assembly code and overall performance. For Petals and Exo, these projects use layer parallelism, which would not use all available computing power when running a single inference.

## Usage

You first need to download a model. It's easiest to `git clone` a model from HuggingFace. This model needs to be in the safetensors format. In this example, the `Llama-2-7b-chat-hf` model has been placed in the `models` directory.

To generate the tokenizer.bin file, run the following:

````bash
cd python
python tokenizer.py --model models/Llama-2-7b-chat-hf --tokenizer-type LLAMA
````

Build with (in the root directory):

````bash
make all
````

Run with:

````bash
OMP_NUM_THREADS=4 bin/localinfer models/Llama-2-7b-chat-hf -i "It was a dark and stormy night" -n 50
````

### Clustered (WIP)

On the workers (nodes):

````bash
bin/worker 1234
````

On the host, edit bin/client.c to enter the workers you're going to use. Then run:

````bash
make all
bin/client models/Llama-2-7b-chat-hf -i "It was a dark and stormy night" -n 100
````

## Development

It's recommended to use `clangd`. `bear` will generate `compile_commands.json` that allow it to function.

````bash
sudo apt install clang libomp-dev bear
make compile_commands
````

## License

Diolkos is licensed AGPLv3+. Note that the original llama2.c is licensed MIT.

    Diolkos - LLM inference cluster software written in C
    Copyright (C) 2023 Andrej Karpathy (llama2.c)
    Copyright (C) 2025 Krol Inventions B.V.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Contributing

Bugfixes are always welcome. For features, open a ticket first. Forking and running experiments with this code is always appreciated though!

Note: please license your contributions as MIT. This way I can switch to MIT later if the AGPLv3 is a hurdle.

## FAQ

#### Why C? Why not Rust?

I first tried forking `lm.rs`, but that was a huge hassle. Rust is great for when you know exactly what you want. Writing the code takes 10x as long, but you get production-ready code on the first try.

This however conflicts with the experimental nature of Diolkos. It's very nice to do something quickly in a hacky way to see if it actually works and if it is what you want. If so, you can secure it later. Besides - Diolkos is not designed to be exposed to the internet anyway.

#### Why C? Why not C++?

C++ has way too many complicated features. Give C a try. Everything makes sense. Some things are a bit more manual and verbose, but with a LLM helping out that's not a problem. Make sure to use plenty of asserts, and don't hesitate to run Valgrind!

#### Why doesn't this use llama.cpp/ggml?

It's easier to experiment when you control and all the code. Maybe more backends will be added in the future.

#### Should I be scared of the AGPL?

The AGPL (in full: Affero GNU Public License) extends the GPL with a clause that requires the following: if you make any changes to the code, you need to make these changes available to the users of the software, even if these users access the software over the network.

This means that as long as you either don't make changes, or place a link somewhere that allows the users to download the changed code - you're good.

(note that this is not legal advice)

#### Can we pay you to add a feature or hire you for consulting?

Yes, see the [Krol Inventions](https://krolinventions.com) website.
