	�Ü Wf@�Ü Wf@!�Ü Wf@	p������?p������?!p������?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�Ü Wf@�bFx�F@1h��l`@A�5�!ݡ?I*�dq���?Y~s��o�?*	j�tC2�@2�
JIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2#N'��-@!���5E�X@)��-ʴ-@1���{%�X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchrݔ�Z	�?!�o2�?��?)rݔ�Z	�?1�o2�?��?:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl�M�E�-@!l)A�X@)����Z��?1������?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelisme�<$�?!=!�&^�?)�CP5z5�?16N���?:Preprocessing2�
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle��ĭ��?!���f�~�?)��ĭ��?1���f�~�?:Preprocessing2F
Iterator::Model�����֯?!�p��?)�f���u?1�j+V��?:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache75�|��-@!��=`�X@)�:��Kt?1��ke���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 25.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9p������?It&� :@Q��"� aR@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�bFx�F@�bFx�F@!�bFx�F@      ��!       "	h��l`@h��l`@!h��l`@*      ��!       2	�5�!ݡ?�5�!ݡ?!�5�!ݡ?:	*�dq���?*�dq���?!*�dq���?B      ��!       J	~s��o�?~s��o�?!~s��o�?R      ��!       Z	~s��o�?~s��o�?!~s��o�?b      ��!       JGPUYp������?b qt&� :@y��"� aR@