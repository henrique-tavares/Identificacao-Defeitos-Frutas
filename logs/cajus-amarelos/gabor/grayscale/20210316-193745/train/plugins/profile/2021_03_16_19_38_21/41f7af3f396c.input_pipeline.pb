	&�5̼L@&�5̼L@!&�5̼L@	&vm|�o@&vm|�o@!&vm|�o@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6&�5̼L@c���&H@1����sI@A�K8��?I4�Op1�?Y1���C~@*	�l�����@2�
JIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2�3���"@!9�` �X@)W�/�'�"@1��V۠X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch�n,(ʤ?!8Mc�0g�?)�n,(ʤ?18Mc�0g�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism���g��?!�<�C#�?)��C�b�?1-,����?:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImplܝ��.�"@!��͒�X@)��s��?1+�BRmr�?:Preprocessing2�
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle�o_��?!c-�_���?)�o_��?1c-�_���?:Preprocessing2F
Iterator::ModelHk:!t�?!�ɢ��?)�b� ��?1�Z1]_b�?:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache��r�m�"@!�tɓ��X@)�����q?1�e��/��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 6.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9'vm|�o@I����@Q��LS$V@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	c���&H@c���&H@!c���&H@      ��!       "	����sI@����sI@!����sI@*      ��!       2	�K8��?�K8��?!�K8��?:	4�Op1�?4�Op1�?!4�Op1�?B      ��!       J	1���C~@1���C~@!1���C~@R      ��!       Z	1���C~@1���C~@!1���C~@b      ��!       JGPUY'vm|�o@b q����@y��LS$V@