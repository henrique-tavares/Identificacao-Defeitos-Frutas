�	iUK:ʑ<@iUK:ʑ<@!iUK:ʑ<@	k����+�?k����+�?!k����+�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6iUK:ʑ<@Ը7�a�9@1mr����?A�Rb���?Ip\�M��?Y%��1�?*	     �Q@2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchEh׿�?!�5W0S�J@)Eh׿�?1�5W0S�J@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism�gB�Ē�?!����R@)����K�?1-?ͷ��2@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCachesePmp"�?! ���1@)��0���?1��$u��(@:Preprocessing2F
Iterator::Model��l#�?!�����T@)�D�<�|?1т��cq#@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl<-?p�'p?!�6��@)<-?p�'p?1�6��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9k����+�?I�"QP�SX@Q2��<98 @Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Ը7�a�9@Ը7�a�9@!Ը7�a�9@      ��!       "	mr����?mr����?!mr����?*      ��!       2	�Rb���?�Rb���?!�Rb���?:	p\�M��?p\�M��?!p\�M��?B      ��!       J	%��1�?%��1�?!%��1�?R      ��!       Z	%��1�?%��1�?!%��1�?b      ��!       JGPUYk����+�?b q�"QP�SX@y2��<98 @�"�
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsϔ�'�׺?!ϔ�'�׺?";
sequential_14/dense_70/SoftmaxSoftmaxQ��ٜ?!2�B�)�?"I
-gradient_tape/sequential_14/dense_70/MatMul_1MatMul�7��0��?!0"��O��?"I
-gradient_tape/sequential_14/dense_69/MatMul_1MatMul������?!a��p�?";
sequential_14/dense_66/MatMulMatMul�3͹�S�?!�J$���?0"I
-gradient_tape/sequential_14/dense_67/MatMul_1MatMul|�CUE�?!X��d,��?"I
+gradient_tape/sequential_14/dense_66/MatMulMatMul�QZl�?!�1鏯P�?0"I
-gradient_tape/sequential_14/dense_68/MatMul_1MatMul�&{�#�?!TK�Ȓj�?";
sequential_14/dense_67/MatMulMatMul%���Fۓ?!f��4G��?0";
sequential_14/dense_68/MatMulMatMul��e�ZƎ?!��
z��?0Q      Y@Y�Cc}(@a����S�U@q�e2i�U@y-S�ұ�?"�
both�Your program is POTENTIALLY input-bound because 90.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�84.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 