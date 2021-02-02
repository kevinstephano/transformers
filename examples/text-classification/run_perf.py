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
parser.add_argument('--te', action='store_true', help='Turn off jit.')
parser.add_argument('--batch_size', default='256', type=str, help='Batch size.')
parser.add_argument('--epochs', default='20.0', type=str, help='Number of epochs.')
parser.add_argument('--fused_adam', action='store_true', help='Use fused adam.  You have to turn this on because APEX might not be installed.')

args = parser.parse_args()

env_args = ['TOKENIZERS_PARALLELISM=true']
if args.print_kernel :
    env_args += ['PYTORCH_NVFUSER_DUMP=cuda_kernel']
if args.nojit :
    env_args += ['PYTORCH_JIT_ENABLE=0']
else :
    env_args += ['PYTORCH_JIT_ENABLE=1']

if args.te :
    env_args += ['PYTORCH_NVFUSER_ENABLE=0']
else :
    env_args += ['PYTORCH_NVFUSER_ENABLE=1']

if args.fused_adam :
    env_args += ['USE_FUSED_ADAM=1']
else :
    env_args += ['USE_FUSED_ADAM=0']

benchmark_cmd = ['python', 'run_glue.py', '--overwrite_output_dir', '--model_name_or_path', 'bert-base-cased', \
                 '--task_name', args.task_name, '--do_train', '--max_seq_length', '128', '--seed', '0',        \
                 '--per_device_train_batch_size', args.batch_size, '--learning_rate', '2e-5',                  \
                 '--num_train_epochs', args.epochs, '--output_dir',  os.getcwd() + '/' + args.task_name ]

prof_prefix = ['nsys', 'nvprof', '--print-gpu-trace']
prof_options  = ['--max_steps', '10' ]
run_options   = ['--do_eval' ]
pad_options = ['--pad_to_max_length']

cmd_list = env_args

if args.profile :
    cmd_list += prof_prefix
cmd_list += benchmark_cmd
if args.pad :
    cmd_list += pad_options
if args.profile or args.ten_steps :
    cmd_list += prof_options
else :
    cmd_list += run_options

print(cmd_list)
cmd_str = ''
for item in cmd_list :
    cmd_str += ' ' + item
subprocess.run(cmd_str, stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True, shell=True, cwd=os.getcwd())
