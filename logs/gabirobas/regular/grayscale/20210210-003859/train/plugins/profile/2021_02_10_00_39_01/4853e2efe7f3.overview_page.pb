�	�Q+L_f@�Q+L_f@!�Q+L_f@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�Q+L_f@|�O��J@1Bt�R^@Atzލ��?I�?�0 �?*	�Zd�T@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch���<,�?!s�����G@)���<,�?1s�����G@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism��3���?!z���L�R@)�^)��?1S/��3;@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache�>��?!�A��w3@)�P��9�?1�P!3�-@:Preprocessing2F
Iterator::Model��2#�?!z��"T@)�TO�}s?1��>��@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl�f+/�o?!�`fB��@)�f+/�o?1�`fB��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 30.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI�K;�;M?@Q
-��,Q@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	|�O��J@|�O��J@!|�O��J@      ��!       "	Bt�R^@Bt�R^@!Bt�R^@*      ��!       2	tzލ��?tzލ��?!tzލ��?:	�?�0 �?�?�0 �?!�?�0 �?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�K;�;M?@y
-��,Q@�"m
Agradient_tape/sequential_22/conv2d_82/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��C5IϿ?!��C5IϿ?0"a
@gradient_tape/sequential_22/max_pooling2d_81/MaxPool/MaxPoolGradMaxPoolGrad���ө?!R}g\�?"k
@gradient_tape/sequential_22/conv2d_82/Conv2D/Conv2DBackpropInputConv2DBackpropInputf���Ƨ?!�è�M�?0"m
Agradient_tape/sequential_22/conv2d_81/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��e�?!���?0"<
sequential_22/conv2d_82/Conv2DConv2DU�I�J�?!�&��|�?0".
Adam/gradients/mul_10Mul��)#�?!LB^�?"4
sequential_22/conv2d_81/mulMul�
3��?!حQ���?"-
Adam/gradients/mul_9Mul�4#X��?!xV��V�?".
Adam/gradients/mul_11Mul���N�?!�L�vL��?"<
sequential_22/conv2d_81/BiasAddBiasAdd���ɑ�?!ܰ"��2�?I���FW?Q�碂�X@Y�:Ӹ�Q@aX�r��W@qI�ȶ�@D@yi���2f?"�

both�Your program is POTENTIALLY input-bound because 30.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�40.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 