�	�����g@�����g@!�����g@	-WX��?-WX��?!-WX��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�����g@�8�#+UF@1����a@A����?I�~�� @Y�� :�?*	+�66�@2�
JIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2�g��Җ3@!�Y�0��X@)�~�7�3@1���t�X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchD6�.6��?!����F�?)D6�.6��?1����F�?:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl����3@!\��$��X@)N��1�M�?1��z1ч�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismL��$wج?!���[
T�?)���V�?1�Aô?:Preprocessing2�
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle@0G��ۄ?!j�,ׁ�?)@0G��ۄ?1j�,ׁ�?:Preprocessing2F
Iterator::Model@mT�Y�?!�1}
��?)��dt?1���p�?:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache b����3@!W΂��X@)[�a/�m?1!�_1
ݒ?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9-WX��?I��:�E�8@Q� E���R@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�8�#+UF@�8�#+UF@!�8�#+UF@      ��!       "	����a@����a@!����a@*      ��!       2	����?����?!����?:	�~�� @�~�� @!�~�� @B      ��!       J	�� :�?�� :�?!�� :�?R      ��!       Z	�� :�?�� :�?!�� :�?b      ��!       JGPUY-WX��?b q��:�E�8@y� E���R@�"l
@gradient_tape/sequential_5/conv2d_21/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterޏ)�3Ӻ?!ޏ)�3Ӻ?0"-
IteratorGetNext/_1_Send$.Z�ƀ�?!��(�)�?"l
@gradient_tape/sequential_5/conv2d_20/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�5�#R�?!4�Ә@�?0";
sequential_5/conv2d_20/Conv2DConv2D��kA��?!�
�ۀ�?0"`
?gradient_tape/sequential_5/max_pooling2d_20/MaxPool/MaxPoolGradMaxPoolGrad��"F�?!'SE�B�?"j
?gradient_tape/sequential_5/conv2d_21/Conv2D/Conv2DBackpropInputConv2DBackpropInputs�6I�Z�?!�z.���?0";
sequential_5/conv2d_21/Conv2DConv2D�o8#� �?!��N�?0"-
Adam/gradients/mul_9Muleo&�zg�?!����?".
Adam/gradients/mul_11MulvP���G�?!߫�f���?".
Adam/gradients/mul_10Mul0���
C�?!"+`0�?IEEW�q�?QF�rDw�X@Y��Ӭ��%@a.�e��AV@qʱO�@y�4G|2�j?"�	
both�Your program is POTENTIALLY input-bound because 23.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 