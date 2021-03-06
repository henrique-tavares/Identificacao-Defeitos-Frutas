�	�)t^c�=@�)t^c�=@!�)t^c�=@	�oO�a�?�oO�a�?!�oO�a�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�)t^c�=@�����6;@1M�d�Q�?A��Im �?I.:Yj��?YI���Σ�?*	��S�KS@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch� �S�D�?!JI(yX�E@)� �S�D�?1JI(yX�E@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache�乾�?!ȍ=��?@)� 4J���?1��S%;�;@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism�{,GȨ?!B�P=[O@)��I`s�?1�����3@:Preprocessing2F
Iterator::ModelkQL� �?!��0�TQ@)6���q?1w��}_{@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl�/��Ch?!V^Mϒ�@)�/��Ch?1V^Mϒ�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�oO�a�?ISU}GoX@Q�X0}�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�����6;@�����6;@!�����6;@      ��!       "	M�d�Q�?M�d�Q�?!M�d�Q�?*      ��!       2	��Im �?��Im �?!��Im �?:	.:Yj��?.:Yj��?!.:Yj��?B      ��!       J	I���Σ�?I���Σ�?!I���Σ�?R      ��!       Z	I���Σ�?I���Σ�?!I���Σ�?b      ��!       JGPUY�oO�a�?b qSU}GoX@y�X0}�?�"�
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsq��uS��?!q��uS��?":
sequential_5/dense_23/SoftmaxSoftmax����q�?!/r�~��?":
sequential_5/dense_21/MatMulMatMul4�T��2�?!��bp�?0"H
,gradient_tape/sequential_5/dense_21/MatMul_1MatMul��~�1֗?!gؔ)k�?"H
*gradient_tape/sequential_5/dense_21/MatMulMatMul&l�A;ŗ?!�e�k�c�?0"H
,gradient_tape/sequential_5/dense_23/MatMul_1MatMult�fD!�?!:B���G�?"H
*gradient_tape/sequential_5/dense_20/MatMulMatMulD���ޖ?!^U���?0":
sequential_5/dense_22/MatMulMatMulQ�R�Nؕ?!���po�?0"H
,gradient_tape/sequential_5/dense_22/MatMul_1MatMul<b�]��?!�Z�����?":
sequential_5/dense_20/MatMulMatMul<b�]��?!�0І��?0Q      Y@Y�{��^�*@a�P�"�U@q%h��V�L@y�$ԙ��?"�
both�Your program is POTENTIALLY input-bound because 91.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�57.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 