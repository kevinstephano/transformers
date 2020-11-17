import subprocess
import os
import argparse
import sys


parser = argparse.ArgumentParser(description='Profile Codegen')
parser.add_argument('--cuda-bin', default='/usr/local/cuda/bin/', type=str, help='Cuda Path')
parser.add_argument('--task-name', default='MRPC', type=str, help='Glue Benchmark task.')
parser.add_argument('--profile', action='store_true', help='Profile run.')
parser.add_argument('--ten_steps', action='store_true', help='Run for just 10 steps.')
parser.add_argument('--pad', action='store_true', help='Do not create dynamic batches.')
parser.add_argument('--print-kernel', action='store_true', help='Print Fused Kernels.')
parser.add_argument('--nojit', action='store_true', help='Turn off jit.')

args = parser.parse_args()

env_args = ['TOKENIZERS_PARALLELISM=true']
if args.print_kernel :
    env_args += ['PYTORCH_NVFUSER_DUMP=cuda_kernel']
if args.nojit :
    env_args += ['PYTORCH_NVFUSER_ENABLE=0']
else :
    env_args += ['PYTORCH_NVFUSER_ENABLE=1']

benchmark_cmd = ['python', 'run_glue.py', '--overwrite_output_dir', '--model_name_or_path', 'bert-base-cased',            \
                 '--task_name', args.task_name, '--do_train', '--max_seq_length', '128', '--seed', '0',                   \
                 '--per_device_train_batch_size', '256', '--learning_rate', '2e-5', '--num_train_epochs', '20.0',         \
                 '--output_dir',  os.getcwd() + '/' + args.task_name ]

prof_prefix = ['nsys', 'nvprof', '--print-gpu-trace']
prof_options  = ['--max_steps', '10' ]
run_options   = ['--do_eval' ]
nopad_options = ['--no-pad_to_max_length']

cmd_list = env_args

if args.profile :
    cmd_list += prof_prefix
cmd_list += benchmark_cmd
if not args.pad :
    cmd_list += nopad_options
if args.profile or args.ten_steps :
    cmd_list += prof_options
else :
    cmd_list += run_options

print(cmd_list)
cmd_str = ''
for item in cmd_list :
    cmd_str += ' ' + item
subprocess.run(cmd_str, stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True, shell=True, cwd=os.getcwd())
