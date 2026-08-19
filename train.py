/kaggle/working/road-detection
W0819 08:44:52.769000 134 torch/distributed/run.py:852] 
W0819 08:44:52.769000 134 torch/distributed/run.py:852] *****************************************
W0819 08:44:52.769000 134 torch/distributed/run.py:852] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0819 08:44:52.769000 134 torch/distributed/run.py:852] *****************************************
[W819 08:44:53.689954787 socket.cpp:207] [c10d] The hostname of the client socket cannot be retrieved. err=-3
[W819 08:44:55.283400161 socket.cpp:207] [c10d] The hostname of the client socket cannot be retrieved. err=-3
[W819 08:44:55.283549248 socket.cpp:207] [c10d] The hostname of the client socket cannot be retrieved. err=-3
Official split: train=1108, val=14
Computing road imbalance: 100%|█████████████| 1108/1108 [00:30<00:00, 36.01it/s]
Road imbalance=19.9660; two-class road weight=2.0000
============================================================================
RepVGG two-stream road training
DDP=True | GPUs=2 | backend=NCCL | AMP=True | channels_last=True
Native crop=1024x1024 | batch/GPU=2 | accumulation=2 | global effective batch=8
Road-aware crop probability=0.70 | minimum coarse road fraction=0.0020
Parameters=13,470,683 | epochs=120 | LR=2.00e-04
Main loss=weighted CE + Dice | auxiliary=centerline BCE + Dice
Distributed native validation: tile=1024, overlap=256, tile batch/GPU=2, Hann blending
============================================================================
/usr/local/lib/python3.12/dist-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
  return func(*args, **kwargs)
[rank0]: Traceback (most recent call last):                                     
[rank0]:   File "/kaggle/working/road-detection/train.py", line 1640, in <module>
[rank0]:     main()
[rank0]:   File "/kaggle/working/road-detection/train.py", line 1511, in main
[rank0]:     train_metrics = train_one_epoch(
[rank0]:                     ^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/road-detection/train.py", line 779, in train_one_epoch
[rank0]:     scaler.scale(scaled_loss).backward()
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/_tensor.py", line 630, in backward
[rank0]:     torch.autograd.backward(
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/autograd/__init__.py", line 364, in backward
[rank0]:     _engine_run_backward(
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py", line 865, in _engine_run_backward
[rank0]:     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: RuntimeError: expect_autograd_hooks_ INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/reducer.cpp":1703, please report a bug to PyTorch. 
[rank1]: Traceback (most recent call last):
[rank1]:   File "/kaggle/working/road-detection/train.py", line 1640, in <module>
[rank1]:     main()
[rank1]:   File "/kaggle/working/road-detection/train.py", line 1511, in main
[rank1]:     train_metrics = train_one_epoch(
[rank1]:                     ^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/road-detection/train.py", line 779, in train_one_epoch
[rank1]:     scaler.scale(scaled_loss).backward()
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/_tensor.py", line 630, in backward
[rank1]:     torch.autograd.backward(
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/autograd/__init__.py", line 364, in backward
[rank1]:     _engine_run_backward(
[rank1]:   File "/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py", line 865, in _engine_run_backward
[rank1]:     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: RuntimeError: expect_autograd_hooks_ INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/reducer.cpp":1703, please report a bug to PyTorch. 
E0819 08:46:02.093000 134 torch/distributed/elastic/multiprocessing/api.py:984] failed (exitcode: 1) local_rank: 0 (pid: 142) of binary: /usr/bin/python3
Traceback (most recent call last):
  File "/usr/local/bin/torchrun", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/run.py", line 991, in main
    run(args)
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/run.py", line 982, in run
    elastic_launch(
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/launcher/api.py", line 170, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/launcher/api.py", line 317, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
train.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2026-08-19_08:46:02
  host      : 094f860e29a5
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 143)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-08-19_08:46:02
  host      : 094f860e29a5
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 142)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
