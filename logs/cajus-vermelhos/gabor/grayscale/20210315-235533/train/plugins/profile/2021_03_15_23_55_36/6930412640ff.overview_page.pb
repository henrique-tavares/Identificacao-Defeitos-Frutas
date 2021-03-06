�	l{�%9�d@l{�%9�d@!l{�%9�d@	�23�I�?�23�I�?!�23�I�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6l{�%9�d@�H���`F@1O�S��]@A�ܶ�Q�?I�>�@Y6\䞮��?*	w���1�@2�
JIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2=����S@!����Q�X@)S[� �G@1��r�X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchj1x���?!�`��d��?)j1x���?1�`��d��?:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImplZ�rLg@!�K�h�X@)����?1�qt"�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismÁ�,`�?!!7˺���?){/�h��?1���I��?:Preprocessing2�
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::ShuffleA�} R��?!|<&����?)A�} R��?1|<&����?:Preprocessing2F
Iterator::ModelE��ذ?!��g^�?)@�z��{u?1�������?:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheND��~j@!�[a���X@)�'eRCk?1�ɅU�z�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 26.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�23�I�?ID�gPV�<@Q��ӓ��Q@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�H���`F@�H���`F@!�H���`F@      ��!       "	O�S��]@O�S��]@!O�S��]@*      ��!       2	�ܶ�Q�?�ܶ�Q�?!�ܶ�Q�?:	�>�@�>�@!�>�@B      ��!       J	6\䞮��?6\䞮��?!6\䞮��?R      ��!       Z	6\䞮��?6\䞮��?!6\䞮��?b      ��!       JGPUY�23�I�?b qD�gPV�<@y��ӓ��Q@�"l
@gradient_tape/sequential_3/conv2d_13/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterz��[횽?!z��[횽?0"`
?gradient_tape/sequential_3/max_pooling2d_12/MaxPool/MaxPoolGradMaxPoolGrad؜k�a��?!��6�?"l
@gradient_tape/sequential_3/conv2d_12/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter����V�?!��i�?0"j
?gradient_tape/sequential_3/conv2d_13/Conv2D/Conv2DBackpropInputConv2DBackpropInputXI��!�?!P#>�Cj�?0";
sequential_3/conv2d_13/Conv2DConv2D$��K��?!���I�?0".
Adam/gradients/mul_18Mul�L/R���?!h� (���?"3
sequential_3/conv2d_12/mulMul�4,���?!&t����?".
Adam/gradients/mul_19Mul�D�4�O�?!���\�?".
Adam/gradients/mul_20Mul<����M�?!^<��:�?";
sequential_3/conv2d_12/BiasAddBiasAddHcu�գ?!dtl�w@�?I�p�%� ~?Q�hS|��X@Y����x$@a���pV@q��uj��3@y��0�&�o?"�

both�Your program is POTENTIALLY input-bound because 26.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�20.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 