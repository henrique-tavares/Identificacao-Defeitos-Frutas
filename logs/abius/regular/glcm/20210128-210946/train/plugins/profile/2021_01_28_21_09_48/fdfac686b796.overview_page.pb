�	��n�V:@��n�V:@!��n�V:@	����9�?����9�?!����9�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��n�V:@b�k_@c8@1��Ù_�?A�k��=�?Id��ST�?Y�hE,b�?*���x�W@)      =2]
&Iterator::Model::MaxIntraOpParallelismt\��J˴?!��8aU@)b��4�8�?1p1VL{N@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch/�HM��?!5DsG~m9@)/�HM��?15DsG~m9@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache��}�<�?!H�r=�#@)�!��u�|?1��Қ��@:Preprocessing2F
Iterator::ModelY��9�?!w��\�V@)y�����q?1�J�_@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl%"���1c?!��$4�@)%"���1c?1��$4�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9����9�?I:&p�tX@Q�S�W�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	b�k_@c8@b�k_@c8@!b�k_@c8@      ��!       "	��Ù_�?��Ù_�?!��Ù_�?*      ��!       2	�k��=�?�k��=�?!�k��=�?:	d��ST�?d��ST�?!d��ST�?B      ��!       J	�hE,b�?�hE,b�?!�hE,b�?R      ��!       Z	�hE,b�?�hE,b�?!�hE,b�?b      ��!       JGPUY����9�?b q:&p�tX@y�S�W�?�"�
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits=�Y^=}�?!=�Y^=}�?"<
sequential_66/dense_255/MatMulMatMula�QU�{�?!��ֹ��?0"<
sequential_66/dense_257/SoftmaxSoftmax�?-M{�?![�~_�]�?"J
.gradient_tape/sequential_66/dense_255/MatMul_1MatMul!�+Ye��?!�P�
or�?"J
.gradient_tape/sequential_66/dense_256/MatMul_1MatMul;(I^���?!�um��h�?"J
.gradient_tape/sequential_66/dense_257/MatMul_1MatMul�I�K�.�?!%��_�N�?"J
,gradient_tape/sequential_66/dense_254/MatMulMatMul����7�?!��/���?0"<
sequential_66/dense_254/MatMulMatMul�a�����?!��?���?0"<
sequential_66/dense_256/MatMulMatMul��i�;Д?!m]��"�?0"J
,gradient_tape/sequential_66/dense_255/MatMulMatMul���B~�?!͘�,�j�?0Q      Y@Y�{��^�*@a�P�"�U@qa�
;sT@y���9��?"�
both�Your program is POTENTIALLY input-bound because 92.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�81.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 