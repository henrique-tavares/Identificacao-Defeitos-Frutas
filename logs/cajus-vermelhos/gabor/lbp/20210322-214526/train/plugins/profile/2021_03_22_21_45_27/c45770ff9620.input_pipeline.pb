	y�'e,A@y�'e,A@!y�'e,A@	߬CH)�?߬CH)�?!߬CH)�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6y�'e,A@�JC��?@1��U�P��?AI���*ݵ?I�xx�B�?Y� �4�?*m���5l@)       =2�
JIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2�=#��?!�R�Ū�J@)�u��=�?1}d�@pH@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchm�i�*��?!xLr}6@)m�i�*��?1xLr}6@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism��TO��?!�)�#1HB@)f�?�C�?1����&,@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl�P����?!	�e���M@)���Kċ?1�����@:Preprocessing2�
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle�B�Գ �?!�s��Uk@)�B�Գ �?1�s��Uk@:Preprocessing2F
Iterator::Model�N�P��?!�I�s�C@):̗`}?1����&4	@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheq��H/j�?!	��o�$N@){�%9`Wc?1�.���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9߬CH)�?I�P~�_X@Q�%�h���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�JC��?@�JC��?@!�JC��?@      ��!       "	��U�P��?��U�P��?!��U�P��?*      ��!       2	I���*ݵ?I���*ݵ?!I���*ݵ?:	�xx�B�?�xx�B�?!�xx�B�?B      ��!       J	� �4�?� �4�?!� �4�?R      ��!       Z	� �4�?� �4�?!� �4�?b      ��!       JGPUY߬CH)�?b q�P~�_X@y�%�h���?