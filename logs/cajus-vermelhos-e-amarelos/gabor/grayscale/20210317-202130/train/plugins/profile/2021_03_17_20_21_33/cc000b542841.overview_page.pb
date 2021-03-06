�	��k�K]j@��k�K]j@!��k�K]j@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��k�K]j@~�p�fH@1��#*�d@A���Ɋ�?I�$����?*	;��v^L�@2�
JIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2C�i�q�@!��[�X@)���j�@1���&أX@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchԁ��V_�?!3�\Vp�?)ԁ��V_�?13�\Vp�?:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl[{�@!����X@)zZ����?1�q��@�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelisml� [��?!'�?j@�?)1	�n�?1Kȍ�'��?:Preprocessing2�
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle.9��?!��̆�?).9��?1��̆�?:Preprocessing2F
Iterator::Modelt��%�?!
�iʒ��?)����u?1O5VD9�?:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache)�QGǵ@!~,k���X@)zm6Vb�e?1S�G����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIH��3i8@Qn�	�e�R@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	~�p�fH@~�p�fH@!~�p�fH@      ��!       "	��#*�d@��#*�d@!��#*�d@*      ��!       2	���Ɋ�?���Ɋ�?!���Ɋ�?:	�$����?�$����?!�$����?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qH��3i8@yn�	�e�R@�"l
@gradient_tape/sequential_7/conv2d_27/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter-���?!-���?0"`
?gradient_tape/sequential_7/max_pooling2d_27/MaxPool/MaxPoolGradMaxPoolGradk��uq�?!�����_�?"-
Adam/gradients/mul_9Mul)�J�a��?!hGy>,�?".
Adam/gradients/mul_10MulD��[��?!9�tU���?"3
sequential_7/conv2d_27/mulMulK�V�1��?!�4ej}%�?".
Adam/gradients/mul_11MulX:(@���?!�;j�v�?"j
?gradient_tape/sequential_7/conv2d_28/Conv2D/Conv2DBackpropInputConv2DBackpropInput�*��߂�?!M!�J��?0"l
@gradient_tape/sequential_7/conv2d_28/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterD��a���?!���T��?0";
sequential_7/conv2d_27/BiasAddBiasAdd3K �?!e�W??�?";
sequential_7/conv2d_27/SigmoidSigmoidH�l��?!�S��?�?I܆��w�t?Q*\!B��X@Y��/��&@a�N�>V@q+&�:&�6@yr��&�f_?"�

both�Your program is POTENTIALLY input-bound because 23.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�23.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 