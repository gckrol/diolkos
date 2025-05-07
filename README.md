# BoardMind

What if that pile of electronics junk you have could think? Now it can! BoardMind turns all your single board computers, old laptops, washing machines (well - not yet) into a cluster that can run inference at acceptable speeds.

- Are you preparing for after the apocalypse, where supply of electronics is limited and you have to use what you got?
- Do you want to play around with electronics and inference, and learn how it works?
- Do you want to reduce your dependence on the big players (Nvidia, AMD, Intel, Apple), and run inference on RISC-V or ARM?

Give BoardMind a try!

BoardMind started as a fork of [llama2.c](https://github.com/karpathy/llama2.c). Highly recommended if you want to hack on some LLM inference code!

## Status

The code is still highly experimental, but this all works:

- Loading of safetensor models (llama2 architecture).
- Automatic quantization to Q8_0.
- Efficient caching/loading of quantized models.
- 8 bit inference at 90% of the speed of Llama.cpp.
- Clustered operation at 90% efficiency.

Supported model types:
- Llama 2

Supported model formats:
- Non quantized (F16/BF16/BF32) safetensor. These are the most common models on HuggingFace.

## Goals

Project goals:
- High quality, plain C code
- Runs on all devices, on Linux, especially tiny ones (RISC-V, ARM boards)
- Uses the CPU, not the GPU, no accelerators.
- Good performance
- Small, predictable RAM usage (no crashes while loading)
- Starts up quickly
- Easy to set up and use
- Production ready (within the constraints of the hardware)
- Easy to hack on

Non goals:
- GPU or other accelerator support - unless it's 2 lines of code.
- Windows/MacOS support

## Running

You first need to download a model. It's easierst to `git clone` a model from HuggingFace. This model needs to be
in hte safetensors format. In this example, the `Llama-2-7b-chat-hf` model has been placed in the `models` directory.

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
OMP_NUM_THREADS=4 bin/plainllm models/Llama-2-7b-chat-hf -i "It was a dark and stormy night" -n 50
````

## License

BoardMind is licensed AGPLv3. Note that the original llama2.c is licenced MIT - and this means the unchanged parts of the code still are.

    BoardMind - LLM inference cluster software written in C
    Copyright (C) 2023 Andrej Karpathy (llama2.c)
    Copyright (C) 2025 Krol Inventions B.V.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Contributing

Bugfixes are always welcome. For features, open a ticket first. Forking and running experiments with this code is always appreciated though!

Note: please licence your contributions as MIT. This way I can switch to MIT later if the AGPLv3 is a hurdle - or licence it under a different licence to companies that are scared of the AGPL. That way I could actually get paid for working on open source sofware.
