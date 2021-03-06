�	�=셂�c@�=셂�c@!�=셂�c@	�q)�dg�?�q)�dg�?!�q)�dg�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�=셂�c@��J\�lF@1?rk�m�[@A�3����?I=����#�?Y������?*	|?5^�b�@2�
JIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2,��f*�@!q�*��X@)��py@1ϔV�(�X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch�ǚ�A�?!�e��/��?)�ǚ�A�?1�e��/��?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism�+��ص�?!����7�?)d �.���?1��n�3b�?:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl��q�@!/�z�S�X@)�Ws�`��?1��$P�p�?:Preprocessing2�
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::ShuffleLǜg�K�?!/Eo��s�?)Lǜg�K�?1/Eo��s�?:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheƋ�!r�@!Ih�Q�X@)t��)|?1�g�m&��?:Preprocessing2F
Iterator::Model������?!��e:���?)��ڦx\t?1:R`gU>�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 28.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�q)�dg�?I�Ҳ���=@Q_�h�~ZQ@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��J\�lF@��J\�lF@!��J\�lF@      ��!       "	?rk�m�[@?rk�m�[@!?rk�m�[@*      ��!       2	�3����?�3����?!�3����?:	=����#�?=����#�?!=����#�?B      ��!       J	������?������?!������?R      ��!       Z	������?������?!������?b      ��!       JGPUY�q)�dg�?b q�Ҳ���=@y_�h�~ZQ@�"l
@gradient_tape/sequential_7/conv2d_28/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter}طƺ��?!}طƺ��?0"`
?gradient_tape/sequential_7/max_pooling2d_27/MaxPool/MaxPoolGradMaxPoolGrad������?!q��a���?"l
@gradient_tape/sequential_7/conv2d_27/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�V�r�?!���EMF�?0".
Adam/gradients/mul_18Mul;nUv�?!��'� �?"3
sequential_7/conv2d_27/mulMul�S�(�?!��6n��?".
Adam/gradients/mul_19Muli����ڦ?!"X�Tȸ�?".
Adam/gradients/mul_20Mul�ك�ڦ?!b�'%��?";
sequential_7/conv2d_27/BiasAddBiasAdd��C)�?!
�G`1�?"-
IteratorGetNext/_1_Send�X����?!8��ە��?"j
?gradient_tape/sequential_7/conv2d_28/Conv2D/Conv2DBackpropInputConv2DBackpropInput����nR�?!�懬�L�?0I�\=�d��?Q��s�X@Y����x$@a���pV@q4��f�5@yHQI�l?"�

both�Your program is POTENTIALLY input-bound because 28.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�21.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 