ќн
Ч
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8нр

conv2d_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_81/kernel
}
$conv2d_81/kernel/Read/ReadVariableOpReadVariableOpconv2d_81/kernel*&
_output_shapes
: *
dtype0
t
conv2d_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_81/bias
m
"conv2d_81/bias/Read/ReadVariableOpReadVariableOpconv2d_81/bias*
_output_shapes
: *
dtype0

conv2d_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_82/kernel
}
$conv2d_82/kernel/Read/ReadVariableOpReadVariableOpconv2d_82/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_82/bias
m
"conv2d_82/bias/Read/ReadVariableOpReadVariableOpconv2d_82/bias*
_output_shapes
: *
dtype0

conv2d_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_83/kernel
}
$conv2d_83/kernel/Read/ReadVariableOpReadVariableOpconv2d_83/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_83/bias
m
"conv2d_83/bias/Read/ReadVariableOpReadVariableOpconv2d_83/bias*
_output_shapes
:@*
dtype0

conv2d_84/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_84/kernel
}
$conv2d_84/kernel/Read/ReadVariableOpReadVariableOpconv2d_84/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_84/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_84/bias
m
"conv2d_84/bias/Read/ReadVariableOpReadVariableOpconv2d_84/bias*
_output_shapes
:@*
dtype0
{
dense_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_96/kernel
t
#dense_96/kernel/Read/ReadVariableOpReadVariableOpdense_96/kernel*
_output_shapes
:	@*
dtype0
s
dense_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_96/bias
l
!dense_96/bias/Read/ReadVariableOpReadVariableOpdense_96/bias*
_output_shapes	
:*
dtype0
|
dense_97/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_97/kernel
u
#dense_97/kernel/Read/ReadVariableOpReadVariableOpdense_97/kernel* 
_output_shapes
:
*
dtype0
s
dense_97/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_97/bias
l
!dense_97/bias/Read/ReadVariableOpReadVariableOpdense_97/bias*
_output_shapes	
:*
dtype0
{
dense_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_98/kernel
t
#dense_98/kernel/Read/ReadVariableOpReadVariableOpdense_98/kernel*
_output_shapes
:	@*
dtype0
r
dense_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_98/bias
k
!dense_98/bias/Read/ReadVariableOpReadVariableOpdense_98/bias*
_output_shapes
:@*
dtype0
z
dense_99/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_99/kernel
s
#dense_99/kernel/Read/ReadVariableOpReadVariableOpdense_99/kernel*
_output_shapes

:@*
dtype0
r
dense_99/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_99/bias
k
!dense_99/bias/Read/ReadVariableOpReadVariableOpdense_99/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0	
l

Variable_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_1
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_81/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_81/kernel/m

+Adam/conv2d_81/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_81/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_81/bias/m
{
)Adam/conv2d_81/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_82/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_82/kernel/m

+Adam/conv2d_82/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_82/kernel/m*&
_output_shapes
:  *
dtype0

Adam/conv2d_82/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_82/bias/m
{
)Adam/conv2d_82/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_82/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_83/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_83/kernel/m

+Adam/conv2d_83/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_83/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_83/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_83/bias/m
{
)Adam/conv2d_83/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_83/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_84/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_84/kernel/m

+Adam/conv2d_84/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_84/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_84/bias/m
{
)Adam/conv2d_84/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_96/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_96/kernel/m

*Adam/dense_96/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_96/kernel/m*
_output_shapes
:	@*
dtype0

Adam/dense_96/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_96/bias/m
z
(Adam/dense_96/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_96/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_97/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_97/kernel/m

*Adam/dense_97/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_97/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_97/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_97/bias/m
z
(Adam/dense_97/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_97/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_98/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_98/kernel/m

*Adam/dense_98/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_98/kernel/m*
_output_shapes
:	@*
dtype0

Adam/dense_98/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_98/bias/m
y
(Adam/dense_98/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_98/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_99/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_99/kernel/m

*Adam/dense_99/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_99/kernel/m*
_output_shapes

:@*
dtype0

Adam/dense_99/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_99/bias/m
y
(Adam/dense_99/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_99/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_81/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_81/kernel/v

+Adam/conv2d_81/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_81/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_81/bias/v
{
)Adam/conv2d_81/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_82/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_82/kernel/v

+Adam/conv2d_82/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_82/kernel/v*&
_output_shapes
:  *
dtype0

Adam/conv2d_82/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_82/bias/v
{
)Adam/conv2d_82/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_82/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_83/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_83/kernel/v

+Adam/conv2d_83/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_83/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_83/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_83/bias/v
{
)Adam/conv2d_83/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_83/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_84/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_84/kernel/v

+Adam/conv2d_84/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_84/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_84/bias/v
{
)Adam/conv2d_84/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_96/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_96/kernel/v

*Adam/dense_96/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_96/kernel/v*
_output_shapes
:	@*
dtype0

Adam/dense_96/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_96/bias/v
z
(Adam/dense_96/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_96/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_97/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_97/kernel/v

*Adam/dense_97/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_97/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_97/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_97/bias/v
z
(Adam/dense_97/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_97/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_98/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_98/kernel/v

*Adam/dense_98/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_98/kernel/v*
_output_shapes
:	@*
dtype0

Adam/dense_98/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_98/bias/v
y
(Adam/dense_98/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_98/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_99/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_99/kernel/v

*Adam/dense_99/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_99/kernel/v*
_output_shapes

:@*
dtype0

Adam/dense_99/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_99/bias/v
y
(Adam/dense_99/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_99/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ьd
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Їd
valuedBd Bd
Ѕ
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures

_rng
	keras_api

_rng
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
R
,regularization_losses
-trainable_variables
.	variables
/	keras_api
h

0kernel
1bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
R
6regularization_losses
7trainable_variables
8	variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
R
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
R
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
R
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
h

Rkernel
Sbias
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
h

Xkernel
Ybias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
R
^regularization_losses
_trainable_variables
`	variables
a	keras_api
h

bkernel
cbias
dregularization_losses
etrainable_variables
f	variables
g	keras_api

hiter

ibeta_1

jbeta_2
	kdecay
llearning_ratemЪmЫ&mЬ'mЭ0mЮ1mЯ:mа;mбLmвMmгRmдSmеXmжYmзbmиcmйvкvл&vм'vн0vо1vп:vр;vсLvтMvуRvфSvхXvцYvчbvшcvщ
 
v
0
1
&2
'3
04
15
:6
;7
L8
M9
R10
S11
X12
Y13
b14
c15
v
0
1
&2
'3
04
15
:6
;7
L8
M9
R10
S11
X12
Y13
b14
c15
­
regularization_losses
mlayer_metrics
trainable_variables

nlayers
ometrics
pnon_trainable_variables
	variables
qlayer_regularization_losses
 

r
_state_var
 

s
_state_var
 
\Z
VARIABLE_VALUEconv2d_81/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_81/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
tlayer_metrics
trainable_variables

ulayers
vmetrics
wnon_trainable_variables
 	variables
xlayer_regularization_losses
 
 
 
­
"regularization_losses
ylayer_metrics
#trainable_variables

zlayers
{metrics
|non_trainable_variables
$	variables
}layer_regularization_losses
\Z
VARIABLE_VALUEconv2d_82/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_82/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
А
(regularization_losses
~layer_metrics
)trainable_variables

layers
metrics
non_trainable_variables
*	variables
 layer_regularization_losses
 
 
 
В
,regularization_losses
layer_metrics
-trainable_variables
layers
metrics
non_trainable_variables
.	variables
 layer_regularization_losses
\Z
VARIABLE_VALUEconv2d_83/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_83/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

00
11

00
11
В
2regularization_losses
layer_metrics
3trainable_variables
layers
metrics
non_trainable_variables
4	variables
 layer_regularization_losses
 
 
 
В
6regularization_losses
layer_metrics
7trainable_variables
layers
metrics
non_trainable_variables
8	variables
 layer_regularization_losses
\Z
VARIABLE_VALUEconv2d_84/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_84/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
В
<regularization_losses
layer_metrics
=trainable_variables
layers
metrics
non_trainable_variables
>	variables
 layer_regularization_losses
 
 
 
В
@regularization_losses
layer_metrics
Atrainable_variables
layers
metrics
non_trainable_variables
B	variables
 layer_regularization_losses
 
 
 
В
Dregularization_losses
layer_metrics
Etrainable_variables
layers
metrics
non_trainable_variables
F	variables
  layer_regularization_losses
 
 
 
В
Hregularization_losses
Ёlayer_metrics
Itrainable_variables
Ђlayers
Ѓmetrics
Єnon_trainable_variables
J	variables
 Ѕlayer_regularization_losses
[Y
VARIABLE_VALUEdense_96/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_96/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
В
Nregularization_losses
Іlayer_metrics
Otrainable_variables
Їlayers
Јmetrics
Љnon_trainable_variables
P	variables
 Њlayer_regularization_losses
[Y
VARIABLE_VALUEdense_97/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_97/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
В
Tregularization_losses
Ћlayer_metrics
Utrainable_variables
Ќlayers
­metrics
Ўnon_trainable_variables
V	variables
 Џlayer_regularization_losses
[Y
VARIABLE_VALUEdense_98/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_98/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
В
Zregularization_losses
Аlayer_metrics
[trainable_variables
Бlayers
Вmetrics
Гnon_trainable_variables
\	variables
 Дlayer_regularization_losses
 
 
 
В
^regularization_losses
Еlayer_metrics
_trainable_variables
Жlayers
Зmetrics
Иnon_trainable_variables
`	variables
 Йlayer_regularization_losses
[Y
VARIABLE_VALUEdense_99/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_99/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

b0
c1

b0
c1
В
dregularization_losses
Кlayer_metrics
etrainable_variables
Лlayers
Мmetrics
Нnon_trainable_variables
f	variables
 Оlayer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16

П0
Р1
 
 
PN
VARIABLE_VALUEVariable2layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUE
Variable_12layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Сtotal

Тcount
У	variables
Ф	keras_api
I

Хtotal

Цcount
Ч
_fn_kwargs
Ш	variables
Щ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

С0
Т1

У	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Х0
Ц1

Ш	variables
}
VARIABLE_VALUEAdam/conv2d_81/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_81/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_82/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_82/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_83/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_83/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_84/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_84/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_96/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_96/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_97/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_97/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_98/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_98/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_99/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_99/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_81/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_81/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_82/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_82/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_83/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_83/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_84/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_84/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_96/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_96/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_97/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_97/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_98/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_98/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_99/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_99/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

$serving_default_random_flip_22_inputPlaceholder*1
_output_shapes
:џџџџџџџџџ*
dtype0*&
shape:џџџџџџџџџ
ъ
StatefulPartitionedCallStatefulPartitionedCall$serving_default_random_flip_22_inputconv2d_81/kernelconv2d_81/biasconv2d_82/kernelconv2d_82/biasconv2d_83/kernelconv2d_83/biasconv2d_84/kernelconv2d_84/biasdense_96/kerneldense_96/biasdense_97/kerneldense_97/biasdense_98/kerneldense_98/biasdense_99/kerneldense_99/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_807360
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ш
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_81/kernel/Read/ReadVariableOp"conv2d_81/bias/Read/ReadVariableOp$conv2d_82/kernel/Read/ReadVariableOp"conv2d_82/bias/Read/ReadVariableOp$conv2d_83/kernel/Read/ReadVariableOp"conv2d_83/bias/Read/ReadVariableOp$conv2d_84/kernel/Read/ReadVariableOp"conv2d_84/bias/Read/ReadVariableOp#dense_96/kernel/Read/ReadVariableOp!dense_96/bias/Read/ReadVariableOp#dense_97/kernel/Read/ReadVariableOp!dense_97/bias/Read/ReadVariableOp#dense_98/kernel/Read/ReadVariableOp!dense_98/bias/Read/ReadVariableOp#dense_99/kernel/Read/ReadVariableOp!dense_99/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_81/kernel/m/Read/ReadVariableOp)Adam/conv2d_81/bias/m/Read/ReadVariableOp+Adam/conv2d_82/kernel/m/Read/ReadVariableOp)Adam/conv2d_82/bias/m/Read/ReadVariableOp+Adam/conv2d_83/kernel/m/Read/ReadVariableOp)Adam/conv2d_83/bias/m/Read/ReadVariableOp+Adam/conv2d_84/kernel/m/Read/ReadVariableOp)Adam/conv2d_84/bias/m/Read/ReadVariableOp*Adam/dense_96/kernel/m/Read/ReadVariableOp(Adam/dense_96/bias/m/Read/ReadVariableOp*Adam/dense_97/kernel/m/Read/ReadVariableOp(Adam/dense_97/bias/m/Read/ReadVariableOp*Adam/dense_98/kernel/m/Read/ReadVariableOp(Adam/dense_98/bias/m/Read/ReadVariableOp*Adam/dense_99/kernel/m/Read/ReadVariableOp(Adam/dense_99/bias/m/Read/ReadVariableOp+Adam/conv2d_81/kernel/v/Read/ReadVariableOp)Adam/conv2d_81/bias/v/Read/ReadVariableOp+Adam/conv2d_82/kernel/v/Read/ReadVariableOp)Adam/conv2d_82/bias/v/Read/ReadVariableOp+Adam/conv2d_83/kernel/v/Read/ReadVariableOp)Adam/conv2d_83/bias/v/Read/ReadVariableOp+Adam/conv2d_84/kernel/v/Read/ReadVariableOp)Adam/conv2d_84/bias/v/Read/ReadVariableOp*Adam/dense_96/kernel/v/Read/ReadVariableOp(Adam/dense_96/bias/v/Read/ReadVariableOp*Adam/dense_97/kernel/v/Read/ReadVariableOp(Adam/dense_97/bias/v/Read/ReadVariableOp*Adam/dense_98/kernel/v/Read/ReadVariableOp(Adam/dense_98/bias/v/Read/ReadVariableOp*Adam/dense_99/kernel/v/Read/ReadVariableOp(Adam/dense_99/bias/v/Read/ReadVariableOpConst*H
TinA
?2=			*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_808281
Ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_81/kernelconv2d_81/biasconv2d_82/kernelconv2d_82/biasconv2d_83/kernelconv2d_83/biasconv2d_84/kernelconv2d_84/biasdense_96/kerneldense_96/biasdense_97/kerneldense_97/biasdense_98/kerneldense_98/biasdense_99/kerneldense_99/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateVariable
Variable_1totalcounttotal_1count_1Adam/conv2d_81/kernel/mAdam/conv2d_81/bias/mAdam/conv2d_82/kernel/mAdam/conv2d_82/bias/mAdam/conv2d_83/kernel/mAdam/conv2d_83/bias/mAdam/conv2d_84/kernel/mAdam/conv2d_84/bias/mAdam/dense_96/kernel/mAdam/dense_96/bias/mAdam/dense_97/kernel/mAdam/dense_97/bias/mAdam/dense_98/kernel/mAdam/dense_98/bias/mAdam/dense_99/kernel/mAdam/dense_99/bias/mAdam/conv2d_81/kernel/vAdam/conv2d_81/bias/vAdam/conv2d_82/kernel/vAdam/conv2d_82/bias/vAdam/conv2d_83/kernel/vAdam/conv2d_83/bias/vAdam/conv2d_84/kernel/vAdam/conv2d_84/bias/vAdam/dense_96/kernel/vAdam/dense_96/bias/vAdam/dense_97/kernel/vAdam/dense_97/bias/vAdam/dense_98/kernel/vAdam/dense_98/bias/vAdam/dense_99/kernel/vAdam/dense_99/bias/v*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_808468сс
п
~
)__inference_dense_99_layer_call_fn_808071

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_8069152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_806379

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Э
Q
5__inference_spatial_dropout2d_22_layer_call_fn_807926

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_8067802
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ї	
н
D__inference_dense_97_layer_call_and_return_conditional_losses_807995

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­B
ъ
I__inference_sequential_22_layer_call_and_return_conditional_losses_806983
random_flip_22_input
conv2d_81_806935
conv2d_81_806937
conv2d_82_806941
conv2d_82_806943
conv2d_83_806947
conv2d_83_806949
conv2d_84_806953
conv2d_84_806955
dense_96_806961
dense_96_806963
dense_97_806966
dense_97_806968
dense_98_806971
dense_98_806973
dense_99_806977
dense_99_806979
identityЂ!conv2d_81/StatefulPartitionedCallЂ!conv2d_82/StatefulPartitionedCallЂ!conv2d_83/StatefulPartitionedCallЂ!conv2d_84/StatefulPartitionedCallЂ dense_96/StatefulPartitionedCallЂ dense_97/StatefulPartitionedCallЂ dense_98/StatefulPartitionedCallЂ dense_99/StatefulPartitionedCallД
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_22_inputconv2d_81_806935conv2d_81_806937*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_81_layer_call_and_return_conditional_losses_8066372#
!conv2d_81/StatefulPartitionedCall
 max_pooling2d_81/PartitionedCallPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_8063432"
 max_pooling2d_81/PartitionedCallЩ
!conv2d_82/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_81/PartitionedCall:output:0conv2d_82_806941conv2d_82_806943*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ§§ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_82_layer_call_and_return_conditional_losses_8066702#
!conv2d_82/StatefulPartitionedCall
 max_pooling2d_82/PartitionedCallPartitionedCall*conv2d_82/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ~~ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_8063552"
 max_pooling2d_82/PartitionedCallЧ
!conv2d_83/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_82/PartitionedCall:output:0conv2d_83_806947conv2d_83_806949*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ||@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_83_layer_call_and_return_conditional_losses_8067032#
!conv2d_83/StatefulPartitionedCall
 max_pooling2d_83/PartitionedCallPartitionedCall*conv2d_83/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_8063672"
 max_pooling2d_83/PartitionedCallЧ
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_83/PartitionedCall:output:0conv2d_84_806953conv2d_84_806955*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_84_layer_call_and_return_conditional_losses_8067362#
!conv2d_84/StatefulPartitionedCall
 max_pooling2d_84/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_8063792"
 max_pooling2d_84/PartitionedCallІ
$spatial_dropout2d_22/PartitionedCallPartitionedCall)max_pooling2d_84/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_8067802&
$spatial_dropout2d_22/PartitionedCallЗ
+global_average_pooling2d_22/PartitionedCallPartitionedCall-spatial_dropout2d_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_22_layer_call_and_return_conditional_losses_8064602-
+global_average_pooling2d_22/PartitionedCallЦ
 dense_96/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_22/PartitionedCall:output:0dense_96_806961dense_96_806963*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_8068042"
 dense_96/StatefulPartitionedCallЛ
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_806966dense_97_806968*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_8068312"
 dense_97/StatefulPartitionedCallК
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_806971dense_98_806973*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_98_layer_call_and_return_conditional_losses_8068582"
 dense_98/StatefulPartitionedCall
dropout_20/PartitionedCallPartitionedCall)dense_98/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_20_layer_call_and_return_conditional_losses_8068912
dropout_20/PartitionedCallД
 dense_99/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0dense_99_806977dense_99_806979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_8069152"
 dense_99/StatefulPartitionedCall
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0"^conv2d_81/StatefulPartitionedCall"^conv2d_82/StatefulPartitionedCall"^conv2d_83/StatefulPartitionedCall"^conv2d_84/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2F
!conv2d_82/StatefulPartitionedCall!conv2d_82/StatefulPartitionedCall2F
!conv2d_83/StatefulPartitionedCall!conv2d_83/StatefulPartitionedCall2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:g c
1
_output_shapes
:џџџџџџџџџ
.
_user_specified_namerandom_flip_22_input

h
L__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_806355

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љі
О
"__inference__traced_restore_808468
file_prefix%
!assignvariableop_conv2d_81_kernel%
!assignvariableop_1_conv2d_81_bias'
#assignvariableop_2_conv2d_82_kernel%
!assignvariableop_3_conv2d_82_bias'
#assignvariableop_4_conv2d_83_kernel%
!assignvariableop_5_conv2d_83_bias'
#assignvariableop_6_conv2d_84_kernel%
!assignvariableop_7_conv2d_84_bias&
"assignvariableop_8_dense_96_kernel$
 assignvariableop_9_dense_96_bias'
#assignvariableop_10_dense_97_kernel%
!assignvariableop_11_dense_97_bias'
#assignvariableop_12_dense_98_kernel%
!assignvariableop_13_dense_98_bias'
#assignvariableop_14_dense_99_kernel%
!assignvariableop_15_dense_99_bias!
assignvariableop_16_adam_iter#
assignvariableop_17_adam_beta_1#
assignvariableop_18_adam_beta_2"
assignvariableop_19_adam_decay*
&assignvariableop_20_adam_learning_rate 
assignvariableop_21_variable"
assignvariableop_22_variable_1
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1/
+assignvariableop_27_adam_conv2d_81_kernel_m-
)assignvariableop_28_adam_conv2d_81_bias_m/
+assignvariableop_29_adam_conv2d_82_kernel_m-
)assignvariableop_30_adam_conv2d_82_bias_m/
+assignvariableop_31_adam_conv2d_83_kernel_m-
)assignvariableop_32_adam_conv2d_83_bias_m/
+assignvariableop_33_adam_conv2d_84_kernel_m-
)assignvariableop_34_adam_conv2d_84_bias_m.
*assignvariableop_35_adam_dense_96_kernel_m,
(assignvariableop_36_adam_dense_96_bias_m.
*assignvariableop_37_adam_dense_97_kernel_m,
(assignvariableop_38_adam_dense_97_bias_m.
*assignvariableop_39_adam_dense_98_kernel_m,
(assignvariableop_40_adam_dense_98_bias_m.
*assignvariableop_41_adam_dense_99_kernel_m,
(assignvariableop_42_adam_dense_99_bias_m/
+assignvariableop_43_adam_conv2d_81_kernel_v-
)assignvariableop_44_adam_conv2d_81_bias_v/
+assignvariableop_45_adam_conv2d_82_kernel_v-
)assignvariableop_46_adam_conv2d_82_bias_v/
+assignvariableop_47_adam_conv2d_83_kernel_v-
)assignvariableop_48_adam_conv2d_83_bias_v/
+assignvariableop_49_adam_conv2d_84_kernel_v-
)assignvariableop_50_adam_conv2d_84_bias_v.
*assignvariableop_51_adam_dense_96_kernel_v,
(assignvariableop_52_adam_dense_96_bias_v.
*assignvariableop_53_adam_dense_97_kernel_v,
(assignvariableop_54_adam_dense_97_bias_v.
*assignvariableop_55_adam_dense_98_kernel_v,
(assignvariableop_56_adam_dense_98_bias_v.
*assignvariableop_57_adam_dense_99_kernel_v,
(assignvariableop_58_adam_dense_99_bias_v
identity_60ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*Ј 
value B <B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB2layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB2layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*
valueB<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesк
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesѓ
№::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*J
dtypes@
>2<			2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_81_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_81_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ј
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_82_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3І
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_82_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ј
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_83_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5І
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_83_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ј
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_84_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7І
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_84_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ї
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_96_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ѕ
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_96_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ћ
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_97_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Љ
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_97_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ћ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_98_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Љ
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_98_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ћ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_99_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Љ
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_99_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16Ѕ
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ї
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ї
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19І
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ў
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21Є
AssignVariableOp_21AssignVariableOpassignvariableop_21_variableIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22І
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ё
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ё
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ѓ
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ѓ
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Г
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_81_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Б
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_81_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Г
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_82_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Б
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_82_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Г
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_83_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Б
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_83_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Г
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_84_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Б
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_84_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35В
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_96_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36А
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_96_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37В
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_97_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38А
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_97_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39В
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_98_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40А
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_98_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41В
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_99_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42А
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_99_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Г
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_81_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Б
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_81_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Г
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_82_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Б
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_82_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Г
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_83_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Б
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_83_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Г
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_84_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Б
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_84_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51В
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_96_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52А
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_96_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53В
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_97_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54А
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_97_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55В
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_98_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56А
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_98_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57В
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_99_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58А
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_99_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_589
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp№

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_59у

Identity_60IdentityIdentity_59:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_60"#
identity_60Identity_60:output:0*
_input_shapesё
ю: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

ј
.__inference_sequential_22_layer_call_fn_807225
random_flip_22_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identityЂStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_8071882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:џџџџџџџџџ:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:g c
1
_output_shapes
:џџџџџџџџџ
.
_user_specified_namerandom_flip_22_input

X
<__inference_global_average_pooling2d_22_layer_call_fn_806466

inputs
identityс
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_22_layer_call_and_return_conditional_losses_8064602
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

e
F__inference_dropout_20_layer_call_and_return_conditional_losses_806886

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
с
~
)__inference_dense_96_layer_call_fn_807984

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_8068042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ќ
р
E__inference_conv2d_84_layer_call_and_return_conditional_losses_807879

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2	
Sigmoidj
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2

IdentityХ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807872*J
_output_shapes8
6:џџџџџџџџџ<<@:џџџџџџџџџ<<@2
	IdentityNЃ

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ<<@2

Identity_1"!

identity_1Identity_1:output:0*6
_input_shapes%
#:џџџџџџџџџ>>@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ>>@
 
_user_specified_nameinputs
ђ

ш
.__inference_sequential_22_layer_call_fn_807313
random_flip_22_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identityЂStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_8072782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:g c
1
_output_shapes
:џџџџџџџџџ
.
_user_specified_namerandom_flip_22_input

р
E__inference_conv2d_81_layer_call_and_return_conditional_losses_807804

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpІ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2	
Sigmoidl
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2

IdentityЩ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807797*N
_output_shapes<
::џџџџџџџџџўў :џџџџџџџџџўў 2
	IdentityNЅ

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџўў 2

Identity_1"!

identity_1Identity_1:output:0*8
_input_shapes'
%:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


*__inference_conv2d_84_layer_call_fn_807888

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_84_layer_call_and_return_conditional_losses_8067362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ<<@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ>>@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ>>@
 
_user_specified_nameinputs
лk
ш

I__inference_sequential_22_layer_call_and_return_conditional_losses_807712

inputs,
(conv2d_81_conv2d_readvariableop_resource-
)conv2d_81_biasadd_readvariableop_resource,
(conv2d_82_conv2d_readvariableop_resource-
)conv2d_82_biasadd_readvariableop_resource,
(conv2d_83_conv2d_readvariableop_resource-
)conv2d_83_biasadd_readvariableop_resource,
(conv2d_84_conv2d_readvariableop_resource-
)conv2d_84_biasadd_readvariableop_resource+
'dense_96_matmul_readvariableop_resource,
(dense_96_biasadd_readvariableop_resource+
'dense_97_matmul_readvariableop_resource,
(dense_97_biasadd_readvariableop_resource+
'dense_98_matmul_readvariableop_resource,
(dense_98_biasadd_readvariableop_resource+
'dense_99_matmul_readvariableop_resource,
(dense_99_biasadd_readvariableop_resource
identityЂ conv2d_81/BiasAdd/ReadVariableOpЂconv2d_81/Conv2D/ReadVariableOpЂ conv2d_82/BiasAdd/ReadVariableOpЂconv2d_82/Conv2D/ReadVariableOpЂ conv2d_83/BiasAdd/ReadVariableOpЂconv2d_83/Conv2D/ReadVariableOpЂ conv2d_84/BiasAdd/ReadVariableOpЂconv2d_84/Conv2D/ReadVariableOpЂdense_96/BiasAdd/ReadVariableOpЂdense_96/MatMul/ReadVariableOpЂdense_97/BiasAdd/ReadVariableOpЂdense_97/MatMul/ReadVariableOpЂdense_98/BiasAdd/ReadVariableOpЂdense_98/MatMul/ReadVariableOpЂdense_99/BiasAdd/ReadVariableOpЂdense_99/MatMul/ReadVariableOpГ
conv2d_81/Conv2D/ReadVariableOpReadVariableOp(conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_81/Conv2D/ReadVariableOpФ
conv2d_81/Conv2DConv2Dinputs'conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў *
paddingVALID*
strides
2
conv2d_81/Conv2DЊ
 conv2d_81/BiasAdd/ReadVariableOpReadVariableOp)conv2d_81_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_81/BiasAdd/ReadVariableOpВ
conv2d_81/BiasAddBiasAddconv2d_81/Conv2D:output:0(conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2
conv2d_81/BiasAdd
conv2d_81/SigmoidSigmoidconv2d_81/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2
conv2d_81/Sigmoid
conv2d_81/mulMulconv2d_81/BiasAdd:output:0conv2d_81/Sigmoid:y:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2
conv2d_81/mul
conv2d_81/IdentityIdentityconv2d_81/mul:z:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2
conv2d_81/Identityё
conv2d_81/IdentityN	IdentityNconv2d_81/mul:z:0conv2d_81/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807633*N
_output_shapes<
::џџџџџџџџџўў :џџџџџџџџџўў 2
conv2d_81/IdentityNЬ
max_pooling2d_81/MaxPoolMaxPoolconv2d_81/IdentityN:output:0*1
_output_shapes
:џџџџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_81/MaxPoolГ
conv2d_82/Conv2D/ReadVariableOpReadVariableOp(conv2d_82_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_82/Conv2D/ReadVariableOpп
conv2d_82/Conv2DConv2D!max_pooling2d_81/MaxPool:output:0'conv2d_82/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ *
paddingVALID*
strides
2
conv2d_82/Conv2DЊ
 conv2d_82/BiasAdd/ReadVariableOpReadVariableOp)conv2d_82_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_82/BiasAdd/ReadVariableOpВ
conv2d_82/BiasAddBiasAddconv2d_82/Conv2D:output:0(conv2d_82/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2
conv2d_82/BiasAdd
conv2d_82/SigmoidSigmoidconv2d_82/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2
conv2d_82/Sigmoid
conv2d_82/mulMulconv2d_82/BiasAdd:output:0conv2d_82/Sigmoid:y:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2
conv2d_82/mul
conv2d_82/IdentityIdentityconv2d_82/mul:z:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2
conv2d_82/Identityё
conv2d_82/IdentityN	IdentityNconv2d_82/mul:z:0conv2d_82/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807646*N
_output_shapes<
::џџџџџџџџџ§§ :џџџџџџџџџ§§ 2
conv2d_82/IdentityNЪ
max_pooling2d_82/MaxPoolMaxPoolconv2d_82/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ~~ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_82/MaxPoolГ
conv2d_83/Conv2D/ReadVariableOpReadVariableOp(conv2d_83_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_83/Conv2D/ReadVariableOpн
conv2d_83/Conv2DConv2D!max_pooling2d_82/MaxPool:output:0'conv2d_83/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ||@*
paddingVALID*
strides
2
conv2d_83/Conv2DЊ
 conv2d_83/BiasAdd/ReadVariableOpReadVariableOp)conv2d_83_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_83/BiasAdd/ReadVariableOpА
conv2d_83/BiasAddBiasAddconv2d_83/Conv2D:output:0(conv2d_83/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2
conv2d_83/BiasAdd
conv2d_83/SigmoidSigmoidconv2d_83/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2
conv2d_83/Sigmoid
conv2d_83/mulMulconv2d_83/BiasAdd:output:0conv2d_83/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2
conv2d_83/mul
conv2d_83/IdentityIdentityconv2d_83/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2
conv2d_83/Identityэ
conv2d_83/IdentityN	IdentityNconv2d_83/mul:z:0conv2d_83/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807659*J
_output_shapes8
6:џџџџџџџџџ||@:џџџџџџџџџ||@2
conv2d_83/IdentityNЪ
max_pooling2d_83/MaxPoolMaxPoolconv2d_83/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ>>@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_83/MaxPoolГ
conv2d_84/Conv2D/ReadVariableOpReadVariableOp(conv2d_84_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_84/Conv2D/ReadVariableOpн
conv2d_84/Conv2DConv2D!max_pooling2d_83/MaxPool:output:0'conv2d_84/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@*
paddingVALID*
strides
2
conv2d_84/Conv2DЊ
 conv2d_84/BiasAdd/ReadVariableOpReadVariableOp)conv2d_84_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_84/BiasAdd/ReadVariableOpА
conv2d_84/BiasAddBiasAddconv2d_84/Conv2D:output:0(conv2d_84/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
conv2d_84/BiasAdd
conv2d_84/SigmoidSigmoidconv2d_84/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
conv2d_84/Sigmoid
conv2d_84/mulMulconv2d_84/BiasAdd:output:0conv2d_84/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
conv2d_84/mul
conv2d_84/IdentityIdentityconv2d_84/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
conv2d_84/Identityэ
conv2d_84/IdentityN	IdentityNconv2d_84/mul:z:0conv2d_84/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807672*J
_output_shapes8
6:џџџџџџџџџ<<@:џџџџџџџџџ<<@2
conv2d_84/IdentityNЪ
max_pooling2d_84/MaxPoolMaxPoolconv2d_84/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_84/MaxPoolЇ
spatial_dropout2d_22/IdentityIdentity!max_pooling2d_84/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
spatial_dropout2d_22/IdentityЙ
2global_average_pooling2d_22/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      24
2global_average_pooling2d_22/Mean/reduction_indicesу
 global_average_pooling2d_22/MeanMean&spatial_dropout2d_22/Identity:output:0;global_average_pooling2d_22/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2"
 global_average_pooling2d_22/MeanЉ
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_96/MatMul/ReadVariableOpВ
dense_96/MatMulMatMul)global_average_pooling2d_22/Mean:output:0&dense_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_96/MatMulЈ
dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_96/BiasAdd/ReadVariableOpІ
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_96/BiasAddt
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_96/ReluЊ
dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_97/MatMul/ReadVariableOpЄ
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_97/MatMulЈ
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_97/BiasAdd/ReadVariableOpІ
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_97/BiasAddt
dense_97/ReluReludense_97/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_97/ReluЉ
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_98/MatMul/ReadVariableOpЃ
dense_98/MatMulMatMuldense_97/Relu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_98/MatMulЇ
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_98/BiasAdd/ReadVariableOpЅ
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_98/BiasAdds
dense_98/ReluReludense_98/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_98/Relu
dropout_20/IdentityIdentitydense_98/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_20/IdentityЈ
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_99/MatMul/ReadVariableOpЄ
dense_99/MatMulMatMuldropout_20/Identity:output:0&dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_99/MatMulЇ
dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_99/BiasAdd/ReadVariableOpЅ
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_99/BiasAdd|
dense_99/SoftmaxSoftmaxdense_99/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_99/Softmax
IdentityIdentitydense_99/Softmax:softmax:0!^conv2d_81/BiasAdd/ReadVariableOp ^conv2d_81/Conv2D/ReadVariableOp!^conv2d_82/BiasAdd/ReadVariableOp ^conv2d_82/Conv2D/ReadVariableOp!^conv2d_83/BiasAdd/ReadVariableOp ^conv2d_83/Conv2D/ReadVariableOp!^conv2d_84/BiasAdd/ReadVariableOp ^conv2d_84/Conv2D/ReadVariableOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp ^dense_98/BiasAdd/ReadVariableOp^dense_98/MatMul/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::2D
 conv2d_81/BiasAdd/ReadVariableOp conv2d_81/BiasAdd/ReadVariableOp2B
conv2d_81/Conv2D/ReadVariableOpconv2d_81/Conv2D/ReadVariableOp2D
 conv2d_82/BiasAdd/ReadVariableOp conv2d_82/BiasAdd/ReadVariableOp2B
conv2d_82/Conv2D/ReadVariableOpconv2d_82/Conv2D/ReadVariableOp2D
 conv2d_83/BiasAdd/ReadVariableOp conv2d_83/BiasAdd/ReadVariableOp2B
conv2d_83/Conv2D/ReadVariableOpconv2d_83/Conv2D/ReadVariableOp2D
 conv2d_84/BiasAdd/ReadVariableOp conv2d_84/BiasAdd/ReadVariableOp2B
conv2d_84/Conv2D/ReadVariableOpconv2d_84/Conv2D/ReadVariableOp2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2B
dense_98/BiasAdd/ReadVariableOpdense_98/BiasAdd/ReadVariableOp2@
dense_98/MatMul/ReadVariableOpdense_98/MatMul/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В
M
1__inference_max_pooling2d_83_layer_call_fn_806373

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_8063672
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
й
n
5__inference_spatial_dropout2d_22_layer_call_fn_807921

inputs
identityЂStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_8067752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѓ
n
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807916

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

o
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_806775

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2і
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeд
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yЯ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ќ
р
E__inference_conv2d_83_layer_call_and_return_conditional_losses_807854

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ||@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2	
Sigmoidj
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2

IdentityХ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807847*J
_output_shapes8
6:џџџџџџџџџ||@:џџџџџџџџџ||@2
	IdentityNЃ

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ||@2

Identity_1"!

identity_1Identity_1:output:0*6
_input_shapes%
#:џџџџџџџџџ~~ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ~~ 
 
_user_specified_nameinputs

G
+__inference_dropout_20_layer_call_fn_808051

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_20_layer_call_and_return_conditional_losses_8068912
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Щ
d
F__inference_dropout_20_layer_call_and_return_conditional_losses_806891

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ё	
н
D__inference_dense_98_layer_call_and_return_conditional_losses_808015

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В
M
1__inference_max_pooling2d_81_layer_call_fn_806349

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_8063432
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

e
F__inference_dropout_20_layer_call_and_return_conditional_losses_808036

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
щ

ъ
.__inference_sequential_22_layer_call_fn_807751

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identityЂStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_8071882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:џџџџџџџџџ:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К
Q
5__inference_spatial_dropout2d_22_layer_call_fn_807964

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_8064502
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


*__inference_conv2d_82_layer_call_fn_807838

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ§§ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_82_layer_call_and_return_conditional_losses_8066702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџџџ 
 
_user_specified_nameinputs
у
~
)__inference_dense_97_layer_call_fn_808004

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_8068312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ
n
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_806780

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Щи
Ж
I__inference_sequential_22_layer_call_and_return_conditional_losses_806932
random_flip_22_input@
<random_rotation_22_stateful_uniform_statefuluniform_resource
conv2d_81_806648
conv2d_81_806650
conv2d_82_806681
conv2d_82_806683
conv2d_83_806714
conv2d_83_806716
conv2d_84_806747
conv2d_84_806749
dense_96_806815
dense_96_806817
dense_97_806842
dense_97_806844
dense_98_806869
dense_98_806871
dense_99_806926
dense_99_806928
identityЂ!conv2d_81/StatefulPartitionedCallЂ!conv2d_82/StatefulPartitionedCallЂ!conv2d_83/StatefulPartitionedCallЂ!conv2d_84/StatefulPartitionedCallЂ dense_96/StatefulPartitionedCallЂ dense_97/StatefulPartitionedCallЂ dense_98/StatefulPartitionedCallЂ dense_99/StatefulPartitionedCallЂ"dropout_20/StatefulPartitionedCallЂ3random_rotation_22/stateful_uniform/StatefulUniformЂ,spatial_dropout2d_22/StatefulPartitionedCallћ
8random_flip_22/random_flip_left_right/control_dependencyIdentityrandom_flip_22_input*
T0*'
_class
loc:@random_flip_22_input*1
_output_shapes
:џџџџџџџџџ2:
8random_flip_22/random_flip_left_right/control_dependencyЫ
+random_flip_22/random_flip_left_right/ShapeShapeArandom_flip_22/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2-
+random_flip_22/random_flip_left_right/ShapeР
9random_flip_22/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9random_flip_22/random_flip_left_right/strided_slice/stackФ
;random_flip_22/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip_22/random_flip_left_right/strided_slice/stack_1Ф
;random_flip_22/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip_22/random_flip_left_right/strided_slice/stack_2Ц
3random_flip_22/random_flip_left_right/strided_sliceStridedSlice4random_flip_22/random_flip_left_right/Shape:output:0Brandom_flip_22/random_flip_left_right/strided_slice/stack:output:0Drandom_flip_22/random_flip_left_right/strided_slice/stack_1:output:0Drandom_flip_22/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3random_flip_22/random_flip_left_right/strided_sliceь
:random_flip_22/random_flip_left_right/random_uniform/shapePack<random_flip_22/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2<
:random_flip_22/random_flip_left_right/random_uniform/shapeЙ
8random_flip_22/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2:
8random_flip_22/random_flip_left_right/random_uniform/minЙ
8random_flip_22/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8random_flip_22/random_flip_left_right/random_uniform/max
Brandom_flip_22/random_flip_left_right/random_uniform/RandomUniformRandomUniformCrandom_flip_22/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02D
Brandom_flip_22/random_flip_left_right/random_uniform/RandomUniformЙ
8random_flip_22/random_flip_left_right/random_uniform/MulMulKrandom_flip_22/random_flip_left_right/random_uniform/RandomUniform:output:0Arandom_flip_22/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2:
8random_flip_22/random_flip_left_right/random_uniform/MulА
5random_flip_22/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5random_flip_22/random_flip_left_right/Reshape/shape/1А
5random_flip_22/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :27
5random_flip_22/random_flip_left_right/Reshape/shape/2А
5random_flip_22/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :27
5random_flip_22/random_flip_left_right/Reshape/shape/3
3random_flip_22/random_flip_left_right/Reshape/shapePack<random_flip_22/random_flip_left_right/strided_slice:output:0>random_flip_22/random_flip_left_right/Reshape/shape/1:output:0>random_flip_22/random_flip_left_right/Reshape/shape/2:output:0>random_flip_22/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:25
3random_flip_22/random_flip_left_right/Reshape/shape
-random_flip_22/random_flip_left_right/ReshapeReshape<random_flip_22/random_flip_left_right/random_uniform/Mul:z:0<random_flip_22/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2/
-random_flip_22/random_flip_left_right/Reshapeе
+random_flip_22/random_flip_left_right/RoundRound6random_flip_22/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2-
+random_flip_22/random_flip_left_right/RoundЖ
4random_flip_22/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:26
4random_flip_22/random_flip_left_right/ReverseV2/axis­
/random_flip_22/random_flip_left_right/ReverseV2	ReverseV2Arandom_flip_22/random_flip_left_right/control_dependency:output:0=random_flip_22/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ21
/random_flip_22/random_flip_left_right/ReverseV2
)random_flip_22/random_flip_left_right/mulMul/random_flip_22/random_flip_left_right/Round:y:08random_flip_22/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2+
)random_flip_22/random_flip_left_right/mul
+random_flip_22/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+random_flip_22/random_flip_left_right/sub/xў
)random_flip_22/random_flip_left_right/subSub4random_flip_22/random_flip_left_right/sub/x:output:0/random_flip_22/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2+
)random_flip_22/random_flip_left_right/sub
+random_flip_22/random_flip_left_right/mul_1Mul-random_flip_22/random_flip_left_right/sub:z:0Arandom_flip_22/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2-
+random_flip_22/random_flip_left_right/mul_1ћ
)random_flip_22/random_flip_left_right/addAddV2-random_flip_22/random_flip_left_right/mul:z:0/random_flip_22/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2+
)random_flip_22/random_flip_left_right/addЃ
5random_flip_22/random_flip_up_down/control_dependencyIdentity-random_flip_22/random_flip_left_right/add:z:0*
T0*<
_class2
0.loc:@random_flip_22/random_flip_left_right/add*1
_output_shapes
:џџџџџџџџџ27
5random_flip_22/random_flip_up_down/control_dependencyТ
(random_flip_22/random_flip_up_down/ShapeShape>random_flip_22/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:2*
(random_flip_22/random_flip_up_down/ShapeК
6random_flip_22/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_flip_22/random_flip_up_down/strided_slice/stackО
8random_flip_22/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip_22/random_flip_up_down/strided_slice/stack_1О
8random_flip_22/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip_22/random_flip_up_down/strided_slice/stack_2Д
0random_flip_22/random_flip_up_down/strided_sliceStridedSlice1random_flip_22/random_flip_up_down/Shape:output:0?random_flip_22/random_flip_up_down/strided_slice/stack:output:0Arandom_flip_22/random_flip_up_down/strided_slice/stack_1:output:0Arandom_flip_22/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_flip_22/random_flip_up_down/strided_sliceу
7random_flip_22/random_flip_up_down/random_uniform/shapePack9random_flip_22/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:29
7random_flip_22/random_flip_up_down/random_uniform/shapeГ
5random_flip_22/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5random_flip_22/random_flip_up_down/random_uniform/minГ
5random_flip_22/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5random_flip_22/random_flip_up_down/random_uniform/max
?random_flip_22/random_flip_up_down/random_uniform/RandomUniformRandomUniform@random_flip_22/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02A
?random_flip_22/random_flip_up_down/random_uniform/RandomUniform­
5random_flip_22/random_flip_up_down/random_uniform/MulMulHrandom_flip_22/random_flip_up_down/random_uniform/RandomUniform:output:0>random_flip_22/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ27
5random_flip_22/random_flip_up_down/random_uniform/MulЊ
2random_flip_22/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip_22/random_flip_up_down/Reshape/shape/1Њ
2random_flip_22/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip_22/random_flip_up_down/Reshape/shape/2Њ
2random_flip_22/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip_22/random_flip_up_down/Reshape/shape/3
0random_flip_22/random_flip_up_down/Reshape/shapePack9random_flip_22/random_flip_up_down/strided_slice:output:0;random_flip_22/random_flip_up_down/Reshape/shape/1:output:0;random_flip_22/random_flip_up_down/Reshape/shape/2:output:0;random_flip_22/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:22
0random_flip_22/random_flip_up_down/Reshape/shape
*random_flip_22/random_flip_up_down/ReshapeReshape9random_flip_22/random_flip_up_down/random_uniform/Mul:z:09random_flip_22/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2,
*random_flip_22/random_flip_up_down/ReshapeЬ
(random_flip_22/random_flip_up_down/RoundRound3random_flip_22/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
(random_flip_22/random_flip_up_down/RoundА
1random_flip_22/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:23
1random_flip_22/random_flip_up_down/ReverseV2/axisЁ
,random_flip_22/random_flip_up_down/ReverseV2	ReverseV2>random_flip_22/random_flip_up_down/control_dependency:output:0:random_flip_22/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2.
,random_flip_22/random_flip_up_down/ReverseV2ј
&random_flip_22/random_flip_up_down/mulMul,random_flip_22/random_flip_up_down/Round:y:05random_flip_22/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2(
&random_flip_22/random_flip_up_down/mul
(random_flip_22/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(random_flip_22/random_flip_up_down/sub/xђ
&random_flip_22/random_flip_up_down/subSub1random_flip_22/random_flip_up_down/sub/x:output:0,random_flip_22/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2(
&random_flip_22/random_flip_up_down/sub
(random_flip_22/random_flip_up_down/mul_1Mul*random_flip_22/random_flip_up_down/sub:z:0>random_flip_22/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2*
(random_flip_22/random_flip_up_down/mul_1я
&random_flip_22/random_flip_up_down/addAddV2*random_flip_22/random_flip_up_down/mul:z:0,random_flip_22/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2(
&random_flip_22/random_flip_up_down/add
random_rotation_22/ShapeShape*random_flip_22/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2
random_rotation_22/Shape
&random_rotation_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_rotation_22/strided_slice/stack
(random_rotation_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice/stack_1
(random_rotation_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice/stack_2д
 random_rotation_22/strided_sliceStridedSlice!random_rotation_22/Shape:output:0/random_rotation_22/strided_slice/stack:output:01random_rotation_22/strided_slice/stack_1:output:01random_rotation_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 random_rotation_22/strided_slice
(random_rotation_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice_1/stackЂ
*random_rotation_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_1/stack_1Ђ
*random_rotation_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_1/stack_2о
"random_rotation_22/strided_slice_1StridedSlice!random_rotation_22/Shape:output:01random_rotation_22/strided_slice_1/stack:output:03random_rotation_22/strided_slice_1/stack_1:output:03random_rotation_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_rotation_22/strided_slice_1
random_rotation_22/CastCast+random_rotation_22/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_22/Cast
(random_rotation_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice_2/stackЂ
*random_rotation_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_2/stack_1Ђ
*random_rotation_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_2/stack_2о
"random_rotation_22/strided_slice_2StridedSlice!random_rotation_22/Shape:output:01random_rotation_22/strided_slice_2/stack:output:03random_rotation_22/strided_slice_2/stack_1:output:03random_rotation_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_rotation_22/strided_slice_2
random_rotation_22/Cast_1Cast+random_rotation_22/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_22/Cast_1З
)random_rotation_22/stateful_uniform/shapePack)random_rotation_22/strided_slice:output:0*
N*
T0*
_output_shapes
:2+
)random_rotation_22/stateful_uniform/shape
'random_rotation_22/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|й Р2)
'random_rotation_22/stateful_uniform/min
'random_rotation_22/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|й @2)
'random_rotation_22/stateful_uniform/maxР
=random_rotation_22/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2?
=random_rotation_22/stateful_uniform/StatefulUniform/algorithmя
3random_rotation_22/stateful_uniform/StatefulUniformStatefulUniform<random_rotation_22_stateful_uniform_statefuluniform_resourceFrandom_rotation_22/stateful_uniform/StatefulUniform/algorithm:output:02random_rotation_22/stateful_uniform/shape:output:0*#
_output_shapes
:џџџџџџџџџ*
shape_dtype025
3random_rotation_22/stateful_uniform/StatefulUniformо
'random_rotation_22/stateful_uniform/subSub0random_rotation_22/stateful_uniform/max:output:00random_rotation_22/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2)
'random_rotation_22/stateful_uniform/subђ
'random_rotation_22/stateful_uniform/mulMul<random_rotation_22/stateful_uniform/StatefulUniform:output:0+random_rotation_22/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_22/stateful_uniform/mulо
#random_rotation_22/stateful_uniformAdd+random_rotation_22/stateful_uniform/mul:z:00random_rotation_22/stateful_uniform/min:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2%
#random_rotation_22/stateful_uniform
(random_rotation_22/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(random_rotation_22/rotation_matrix/sub/yЪ
&random_rotation_22/rotation_matrix/subSubrandom_rotation_22/Cast_1:y:01random_rotation_22/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2(
&random_rotation_22/rotation_matrix/subЎ
&random_rotation_22/rotation_matrix/CosCos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/Cos
*random_rotation_22/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_1/yа
(random_rotation_22/rotation_matrix/sub_1Subrandom_rotation_22/Cast_1:y:03random_rotation_22/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_1п
&random_rotation_22/rotation_matrix/mulMul*random_rotation_22/rotation_matrix/Cos:y:0,random_rotation_22/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/mulЎ
&random_rotation_22/rotation_matrix/SinSin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/Sin
*random_rotation_22/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_2/yЮ
(random_rotation_22/rotation_matrix/sub_2Subrandom_rotation_22/Cast:y:03random_rotation_22/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_2у
(random_rotation_22/rotation_matrix/mul_1Mul*random_rotation_22/rotation_matrix/Sin:y:0,random_rotation_22/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/mul_1у
(random_rotation_22/rotation_matrix/sub_3Sub*random_rotation_22/rotation_matrix/mul:z:0,random_rotation_22/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/sub_3у
(random_rotation_22/rotation_matrix/sub_4Sub*random_rotation_22/rotation_matrix/sub:z:0,random_rotation_22/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/sub_4Ё
,random_rotation_22/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,random_rotation_22/rotation_matrix/truediv/yі
*random_rotation_22/rotation_matrix/truedivRealDiv,random_rotation_22/rotation_matrix/sub_4:z:05random_rotation_22/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2,
*random_rotation_22/rotation_matrix/truediv
*random_rotation_22/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_5/yЮ
(random_rotation_22/rotation_matrix/sub_5Subrandom_rotation_22/Cast:y:03random_rotation_22/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_5В
(random_rotation_22/rotation_matrix/Sin_1Sin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Sin_1
*random_rotation_22/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_6/yа
(random_rotation_22/rotation_matrix/sub_6Subrandom_rotation_22/Cast_1:y:03random_rotation_22/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_6х
(random_rotation_22/rotation_matrix/mul_2Mul,random_rotation_22/rotation_matrix/Sin_1:y:0,random_rotation_22/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/mul_2В
(random_rotation_22/rotation_matrix/Cos_1Cos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Cos_1
*random_rotation_22/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_7/yЮ
(random_rotation_22/rotation_matrix/sub_7Subrandom_rotation_22/Cast:y:03random_rotation_22/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_7х
(random_rotation_22/rotation_matrix/mul_3Mul,random_rotation_22/rotation_matrix/Cos_1:y:0,random_rotation_22/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/mul_3у
&random_rotation_22/rotation_matrix/addAddV2,random_rotation_22/rotation_matrix/mul_2:z:0,random_rotation_22/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/addу
(random_rotation_22/rotation_matrix/sub_8Sub,random_rotation_22/rotation_matrix/sub_5:z:0*random_rotation_22/rotation_matrix/add:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/sub_8Ѕ
.random_rotation_22/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.random_rotation_22/rotation_matrix/truediv_1/yќ
,random_rotation_22/rotation_matrix/truediv_1RealDiv,random_rotation_22/rotation_matrix/sub_8:z:07random_rotation_22/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2.
,random_rotation_22/rotation_matrix/truediv_1Ћ
(random_rotation_22/rotation_matrix/ShapeShape'random_rotation_22/stateful_uniform:z:0*
T0*
_output_shapes
:2*
(random_rotation_22/rotation_matrix/ShapeК
6random_rotation_22/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_rotation_22/rotation_matrix/strided_slice/stackО
8random_rotation_22/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_22/rotation_matrix/strided_slice/stack_1О
8random_rotation_22/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_22/rotation_matrix/strided_slice/stack_2Д
0random_rotation_22/rotation_matrix/strided_sliceStridedSlice1random_rotation_22/rotation_matrix/Shape:output:0?random_rotation_22/rotation_matrix/strided_slice/stack:output:0Arandom_rotation_22/rotation_matrix/strided_slice/stack_1:output:0Arandom_rotation_22/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_rotation_22/rotation_matrix/strided_sliceВ
(random_rotation_22/rotation_matrix/Cos_2Cos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Cos_2Х
8random_rotation_22/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_1/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_1/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_1/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_1StridedSlice,random_rotation_22/rotation_matrix/Cos_2:y:0Arandom_rotation_22/rotation_matrix/strided_slice_1/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_1/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_1В
(random_rotation_22/rotation_matrix/Sin_2Sin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Sin_2Х
8random_rotation_22/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_2/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_2/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_2/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_2StridedSlice,random_rotation_22/rotation_matrix/Sin_2:y:0Arandom_rotation_22/rotation_matrix/strided_slice_2/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_2/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_2Ц
&random_rotation_22/rotation_matrix/NegNeg;random_rotation_22/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/NegХ
8random_rotation_22/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_3/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_3/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_3/stack_2ы
2random_rotation_22/rotation_matrix/strided_slice_3StridedSlice.random_rotation_22/rotation_matrix/truediv:z:0Arandom_rotation_22/rotation_matrix/strided_slice_3/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_3/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_3В
(random_rotation_22/rotation_matrix/Sin_3Sin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Sin_3Х
8random_rotation_22/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_4/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_4/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_4/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_4StridedSlice,random_rotation_22/rotation_matrix/Sin_3:y:0Arandom_rotation_22/rotation_matrix/strided_slice_4/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_4/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_4В
(random_rotation_22/rotation_matrix/Cos_3Cos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Cos_3Х
8random_rotation_22/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_5/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_5/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_5/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_5StridedSlice,random_rotation_22/rotation_matrix/Cos_3:y:0Arandom_rotation_22/rotation_matrix/strided_slice_5/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_5/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_5Х
8random_rotation_22/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_6/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_6/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_6/stack_2э
2random_rotation_22/rotation_matrix/strided_slice_6StridedSlice0random_rotation_22/rotation_matrix/truediv_1:z:0Arandom_rotation_22/rotation_matrix/strided_slice_6/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_6/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_6Ђ
.random_rotation_22/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation_22/rotation_matrix/zeros/mul/yј
,random_rotation_22/rotation_matrix/zeros/mulMul9random_rotation_22/rotation_matrix/strided_slice:output:07random_rotation_22/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2.
,random_rotation_22/rotation_matrix/zeros/mulЅ
/random_rotation_22/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш21
/random_rotation_22/rotation_matrix/zeros/Less/yѓ
-random_rotation_22/rotation_matrix/zeros/LessLess0random_rotation_22/rotation_matrix/zeros/mul:z:08random_rotation_22/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2/
-random_rotation_22/rotation_matrix/zeros/LessЈ
1random_rotation_22/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :23
1random_rotation_22/rotation_matrix/zeros/packed/1
/random_rotation_22/rotation_matrix/zeros/packedPack9random_rotation_22/rotation_matrix/strided_slice:output:0:random_rotation_22/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:21
/random_rotation_22/rotation_matrix/zeros/packedЅ
.random_rotation_22/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.random_rotation_22/rotation_matrix/zeros/Const
(random_rotation_22/rotation_matrix/zerosFill8random_rotation_22/rotation_matrix/zeros/packed:output:07random_rotation_22/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/zerosЂ
.random_rotation_22/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation_22/rotation_matrix/concat/axisц
)random_rotation_22/rotation_matrix/concatConcatV2;random_rotation_22/rotation_matrix/strided_slice_1:output:0*random_rotation_22/rotation_matrix/Neg:y:0;random_rotation_22/rotation_matrix/strided_slice_3:output:0;random_rotation_22/rotation_matrix/strided_slice_4:output:0;random_rotation_22/rotation_matrix/strided_slice_5:output:0;random_rotation_22/rotation_matrix/strided_slice_6:output:01random_rotation_22/rotation_matrix/zeros:output:07random_rotation_22/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2+
)random_rotation_22/rotation_matrix/concatЂ
"random_rotation_22/transform/ShapeShape*random_flip_22/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2$
"random_rotation_22/transform/ShapeЎ
0random_rotation_22/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0random_rotation_22/transform/strided_slice/stackВ
2random_rotation_22/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2random_rotation_22/transform/strided_slice/stack_1В
2random_rotation_22/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2random_rotation_22/transform/strided_slice/stack_2ќ
*random_rotation_22/transform/strided_sliceStridedSlice+random_rotation_22/transform/Shape:output:09random_rotation_22/transform/strided_slice/stack:output:0;random_rotation_22/transform/strided_slice/stack_1:output:0;random_rotation_22/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*random_rotation_22/transform/strided_slice
'random_rotation_22/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_rotation_22/transform/fill_valueЬ
7random_rotation_22/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3*random_flip_22/random_flip_up_down/add:z:02random_rotation_22/rotation_matrix/concat:output:03random_rotation_22/transform/strided_slice:output:00random_rotation_22/transform/fill_value:output:0*1
_output_shapes
:џџџџџџџџџ*
dtype0*
	fill_mode	NEAREST*
interpolation
BILINEAR29
7random_rotation_22/transform/ImageProjectiveTransformV3ь
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCallLrandom_rotation_22/transform/ImageProjectiveTransformV3:transformed_images:0conv2d_81_806648conv2d_81_806650*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_81_layer_call_and_return_conditional_losses_8066372#
!conv2d_81/StatefulPartitionedCall
 max_pooling2d_81/PartitionedCallPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_8063432"
 max_pooling2d_81/PartitionedCallЩ
!conv2d_82/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_81/PartitionedCall:output:0conv2d_82_806681conv2d_82_806683*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ§§ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_82_layer_call_and_return_conditional_losses_8066702#
!conv2d_82/StatefulPartitionedCall
 max_pooling2d_82/PartitionedCallPartitionedCall*conv2d_82/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ~~ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_8063552"
 max_pooling2d_82/PartitionedCallЧ
!conv2d_83/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_82/PartitionedCall:output:0conv2d_83_806714conv2d_83_806716*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ||@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_83_layer_call_and_return_conditional_losses_8067032#
!conv2d_83/StatefulPartitionedCall
 max_pooling2d_83/PartitionedCallPartitionedCall*conv2d_83/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_8063672"
 max_pooling2d_83/PartitionedCallЧ
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_83/PartitionedCall:output:0conv2d_84_806747conv2d_84_806749*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_84_layer_call_and_return_conditional_losses_8067362#
!conv2d_84/StatefulPartitionedCall
 max_pooling2d_84/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_8063792"
 max_pooling2d_84/PartitionedCallО
,spatial_dropout2d_22/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_84/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_8067752.
,spatial_dropout2d_22/StatefulPartitionedCallП
+global_average_pooling2d_22/PartitionedCallPartitionedCall5spatial_dropout2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_22_layer_call_and_return_conditional_losses_8064602-
+global_average_pooling2d_22/PartitionedCallЦ
 dense_96/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_22/PartitionedCall:output:0dense_96_806815dense_96_806817*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_8068042"
 dense_96/StatefulPartitionedCallЛ
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_806842dense_97_806844*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_8068312"
 dense_97/StatefulPartitionedCallК
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_806869dense_98_806871*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_98_layer_call_and_return_conditional_losses_8068582"
 dense_98/StatefulPartitionedCallЧ
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall)dense_98/StatefulPartitionedCall:output:0-^spatial_dropout2d_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_20_layer_call_and_return_conditional_losses_8068862$
"dropout_20/StatefulPartitionedCallМ
 dense_99/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0dense_99_806926dense_99_806928*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_8069152"
 dense_99/StatefulPartitionedCallЃ
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0"^conv2d_81/StatefulPartitionedCall"^conv2d_82/StatefulPartitionedCall"^conv2d_83/StatefulPartitionedCall"^conv2d_84/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall4^random_rotation_22/stateful_uniform/StatefulUniform-^spatial_dropout2d_22/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:џџџџџџџџџ:::::::::::::::::2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2F
!conv2d_82/StatefulPartitionedCall!conv2d_82/StatefulPartitionedCall2F
!conv2d_83/StatefulPartitionedCall!conv2d_83/StatefulPartitionedCall2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2j
3random_rotation_22/stateful_uniform/StatefulUniform3random_rotation_22/stateful_uniform/StatefulUniform2\
,spatial_dropout2d_22/StatefulPartitionedCall,spatial_dropout2d_22/StatefulPartitionedCall:g c
1
_output_shapes
:џџџџџџџџџ
.
_user_specified_namerandom_flip_22_input
Ј
o
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_806440

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2і
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeд
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yЯ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Х
n
5__inference_spatial_dropout2d_22_layer_call_fn_807959

inputs
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_8064402
StatefulPartitionedCallБ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї	
н
D__inference_dense_97_layer_call_and_return_conditional_losses_806831

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѕ
d
+__inference_dropout_20_layer_call_fn_808046

inputs
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_20_layer_call_and_return_conditional_losses_8068862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
B
м
I__inference_sequential_22_layer_call_and_return_conditional_losses_807278

inputs
conv2d_81_807230
conv2d_81_807232
conv2d_82_807236
conv2d_82_807238
conv2d_83_807242
conv2d_83_807244
conv2d_84_807248
conv2d_84_807250
dense_96_807256
dense_96_807258
dense_97_807261
dense_97_807263
dense_98_807266
dense_98_807268
dense_99_807272
dense_99_807274
identityЂ!conv2d_81/StatefulPartitionedCallЂ!conv2d_82/StatefulPartitionedCallЂ!conv2d_83/StatefulPartitionedCallЂ!conv2d_84/StatefulPartitionedCallЂ dense_96/StatefulPartitionedCallЂ dense_97/StatefulPartitionedCallЂ dense_98/StatefulPartitionedCallЂ dense_99/StatefulPartitionedCallІ
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_81_807230conv2d_81_807232*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_81_layer_call_and_return_conditional_losses_8066372#
!conv2d_81/StatefulPartitionedCall
 max_pooling2d_81/PartitionedCallPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_8063432"
 max_pooling2d_81/PartitionedCallЩ
!conv2d_82/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_81/PartitionedCall:output:0conv2d_82_807236conv2d_82_807238*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ§§ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_82_layer_call_and_return_conditional_losses_8066702#
!conv2d_82/StatefulPartitionedCall
 max_pooling2d_82/PartitionedCallPartitionedCall*conv2d_82/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ~~ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_8063552"
 max_pooling2d_82/PartitionedCallЧ
!conv2d_83/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_82/PartitionedCall:output:0conv2d_83_807242conv2d_83_807244*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ||@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_83_layer_call_and_return_conditional_losses_8067032#
!conv2d_83/StatefulPartitionedCall
 max_pooling2d_83/PartitionedCallPartitionedCall*conv2d_83/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_8063672"
 max_pooling2d_83/PartitionedCallЧ
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_83/PartitionedCall:output:0conv2d_84_807248conv2d_84_807250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_84_layer_call_and_return_conditional_losses_8067362#
!conv2d_84/StatefulPartitionedCall
 max_pooling2d_84/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_8063792"
 max_pooling2d_84/PartitionedCallІ
$spatial_dropout2d_22/PartitionedCallPartitionedCall)max_pooling2d_84/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_8067802&
$spatial_dropout2d_22/PartitionedCallЗ
+global_average_pooling2d_22/PartitionedCallPartitionedCall-spatial_dropout2d_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_22_layer_call_and_return_conditional_losses_8064602-
+global_average_pooling2d_22/PartitionedCallЦ
 dense_96/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_22/PartitionedCall:output:0dense_96_807256dense_96_807258*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_8068042"
 dense_96/StatefulPartitionedCallЛ
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_807261dense_97_807263*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_8068312"
 dense_97/StatefulPartitionedCallК
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_807266dense_98_807268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_98_layer_call_and_return_conditional_losses_8068582"
 dense_98/StatefulPartitionedCall
dropout_20/PartitionedCallPartitionedCall)dense_98/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_20_layer_call_and_return_conditional_losses_8068912
dropout_20/PartitionedCallД
 dense_99/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0dense_99_807272dense_99_807274*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_8069152"
 dense_99/StatefulPartitionedCall
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0"^conv2d_81/StatefulPartitionedCall"^conv2d_82/StatefulPartitionedCall"^conv2d_83/StatefulPartitionedCall"^conv2d_84/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2F
!conv2d_82/StatefulPartitionedCall!conv2d_82/StatefulPartitionedCall2F
!conv2d_83/StatefulPartitionedCall!conv2d_83/StatefulPartitionedCall2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ё	
н
D__inference_dense_98_layer_call_and_return_conditional_losses_806858

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і	
н
D__inference_dense_99_layer_call_and_return_conditional_losses_808062

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

o
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807911

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2і
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeд
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yЯ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
нu

__inference__traced_save_808281
file_prefix/
+savev2_conv2d_81_kernel_read_readvariableop-
)savev2_conv2d_81_bias_read_readvariableop/
+savev2_conv2d_82_kernel_read_readvariableop-
)savev2_conv2d_82_bias_read_readvariableop/
+savev2_conv2d_83_kernel_read_readvariableop-
)savev2_conv2d_83_bias_read_readvariableop/
+savev2_conv2d_84_kernel_read_readvariableop-
)savev2_conv2d_84_bias_read_readvariableop.
*savev2_dense_96_kernel_read_readvariableop,
(savev2_dense_96_bias_read_readvariableop.
*savev2_dense_97_kernel_read_readvariableop,
(savev2_dense_97_bias_read_readvariableop.
*savev2_dense_98_kernel_read_readvariableop,
(savev2_dense_98_bias_read_readvariableop.
*savev2_dense_99_kernel_read_readvariableop,
(savev2_dense_99_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop'
#savev2_variable_read_readvariableop	)
%savev2_variable_1_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_81_kernel_m_read_readvariableop4
0savev2_adam_conv2d_81_bias_m_read_readvariableop6
2savev2_adam_conv2d_82_kernel_m_read_readvariableop4
0savev2_adam_conv2d_82_bias_m_read_readvariableop6
2savev2_adam_conv2d_83_kernel_m_read_readvariableop4
0savev2_adam_conv2d_83_bias_m_read_readvariableop6
2savev2_adam_conv2d_84_kernel_m_read_readvariableop4
0savev2_adam_conv2d_84_bias_m_read_readvariableop5
1savev2_adam_dense_96_kernel_m_read_readvariableop3
/savev2_adam_dense_96_bias_m_read_readvariableop5
1savev2_adam_dense_97_kernel_m_read_readvariableop3
/savev2_adam_dense_97_bias_m_read_readvariableop5
1savev2_adam_dense_98_kernel_m_read_readvariableop3
/savev2_adam_dense_98_bias_m_read_readvariableop5
1savev2_adam_dense_99_kernel_m_read_readvariableop3
/savev2_adam_dense_99_bias_m_read_readvariableop6
2savev2_adam_conv2d_81_kernel_v_read_readvariableop4
0savev2_adam_conv2d_81_bias_v_read_readvariableop6
2savev2_adam_conv2d_82_kernel_v_read_readvariableop4
0savev2_adam_conv2d_82_bias_v_read_readvariableop6
2savev2_adam_conv2d_83_kernel_v_read_readvariableop4
0savev2_adam_conv2d_83_bias_v_read_readvariableop6
2savev2_adam_conv2d_84_kernel_v_read_readvariableop4
0savev2_adam_conv2d_84_bias_v_read_readvariableop5
1savev2_adam_dense_96_kernel_v_read_readvariableop3
/savev2_adam_dense_96_bias_v_read_readvariableop5
1savev2_adam_dense_97_kernel_v_read_readvariableop3
/savev2_adam_dense_97_bias_v_read_readvariableop5
1savev2_adam_dense_98_kernel_v_read_readvariableop3
/savev2_adam_dense_98_bias_v_read_readvariableop5
1savev2_adam_dense_99_kernel_v_read_readvariableop3
/savev2_adam_dense_99_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*Ј 
value B <B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB2layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB2layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*
valueB<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЁ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_81_kernel_read_readvariableop)savev2_conv2d_81_bias_read_readvariableop+savev2_conv2d_82_kernel_read_readvariableop)savev2_conv2d_82_bias_read_readvariableop+savev2_conv2d_83_kernel_read_readvariableop)savev2_conv2d_83_bias_read_readvariableop+savev2_conv2d_84_kernel_read_readvariableop)savev2_conv2d_84_bias_read_readvariableop*savev2_dense_96_kernel_read_readvariableop(savev2_dense_96_bias_read_readvariableop*savev2_dense_97_kernel_read_readvariableop(savev2_dense_97_bias_read_readvariableop*savev2_dense_98_kernel_read_readvariableop(savev2_dense_98_bias_read_readvariableop*savev2_dense_99_kernel_read_readvariableop(savev2_dense_99_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_81_kernel_m_read_readvariableop0savev2_adam_conv2d_81_bias_m_read_readvariableop2savev2_adam_conv2d_82_kernel_m_read_readvariableop0savev2_adam_conv2d_82_bias_m_read_readvariableop2savev2_adam_conv2d_83_kernel_m_read_readvariableop0savev2_adam_conv2d_83_bias_m_read_readvariableop2savev2_adam_conv2d_84_kernel_m_read_readvariableop0savev2_adam_conv2d_84_bias_m_read_readvariableop1savev2_adam_dense_96_kernel_m_read_readvariableop/savev2_adam_dense_96_bias_m_read_readvariableop1savev2_adam_dense_97_kernel_m_read_readvariableop/savev2_adam_dense_97_bias_m_read_readvariableop1savev2_adam_dense_98_kernel_m_read_readvariableop/savev2_adam_dense_98_bias_m_read_readvariableop1savev2_adam_dense_99_kernel_m_read_readvariableop/savev2_adam_dense_99_bias_m_read_readvariableop2savev2_adam_conv2d_81_kernel_v_read_readvariableop0savev2_adam_conv2d_81_bias_v_read_readvariableop2savev2_adam_conv2d_82_kernel_v_read_readvariableop0savev2_adam_conv2d_82_bias_v_read_readvariableop2savev2_adam_conv2d_83_kernel_v_read_readvariableop0savev2_adam_conv2d_83_bias_v_read_readvariableop2savev2_adam_conv2d_84_kernel_v_read_readvariableop0savev2_adam_conv2d_84_bias_v_read_readvariableop1savev2_adam_dense_96_kernel_v_read_readvariableop/savev2_adam_dense_96_bias_v_read_readvariableop1savev2_adam_dense_97_kernel_v_read_readvariableop/savev2_adam_dense_97_bias_v_read_readvariableop1savev2_adam_dense_98_kernel_v_read_readvariableop/savev2_adam_dense_98_bias_v_read_readvariableop1savev2_adam_dense_99_kernel_v_read_readvariableop/savev2_adam_dense_99_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *J
dtypes@
>2<			2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Љ
_input_shapes
: : : :  : : @:@:@@:@:	@::
::	@:@:@:: : : : : ::: : : : : : :  : : @:@:@@:@:	@::
::	@:@:@:: : :  : : @:@:@@:@:	@::
::	@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:%	!

_output_shapes
:	@:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :, (
&
_output_shapes
: @: !

_output_shapes
:@:,"(
&
_output_shapes
:@@: #

_output_shapes
:@:%$!

_output_shapes
:	@:!%

_output_shapes	
::&&"
 
_output_shapes
:
:!'

_output_shapes	
::%(!

_output_shapes
:	@: )

_output_shapes
:@:$* 

_output_shapes

:@: +

_output_shapes
::,,(
&
_output_shapes
: : -

_output_shapes
: :,.(
&
_output_shapes
:  : /

_output_shapes
: :,0(
&
_output_shapes
: @: 1

_output_shapes
:@:,2(
&
_output_shapes
:@@: 3

_output_shapes
:@:%4!

_output_shapes
:	@:!5

_output_shapes	
::&6"
 
_output_shapes
:
:!7

_output_shapes	
::%8!

_output_shapes
:	@: 9

_output_shapes
:@:$: 

_output_shapes

:@: ;

_output_shapes
::<

_output_shapes
: 
Р

о
$__inference_signature_wrapper_807360
random_flip_22_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_8063372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:g c
1
_output_shapes
:џџџџџџџџџ
.
_user_specified_namerandom_flip_22_input
ќ
р
E__inference_conv2d_84_layer_call_and_return_conditional_losses_806736

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2	
Sigmoidj
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2

IdentityХ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-806729*J
_output_shapes8
6:џџџџџџџџџ<<@:џџџџџџџџџ<<@2
	IdentityNЃ

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ<<@2

Identity_1"!

identity_1Identity_1:output:0*6
_input_shapes%
#:џџџџџџџџџ>>@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ>>@
 
_user_specified_nameinputs
є	
н
D__inference_dense_96_layer_call_and_return_conditional_losses_807975

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


*__inference_conv2d_81_layer_call_fn_807813

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_81_layer_call_and_return_conditional_losses_8066372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџўў 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
и
Ј
I__inference_sequential_22_layer_call_and_return_conditional_losses_807188

inputs@
<random_rotation_22_stateful_uniform_statefuluniform_resource
conv2d_81_807140
conv2d_81_807142
conv2d_82_807146
conv2d_82_807148
conv2d_83_807152
conv2d_83_807154
conv2d_84_807158
conv2d_84_807160
dense_96_807166
dense_96_807168
dense_97_807171
dense_97_807173
dense_98_807176
dense_98_807178
dense_99_807182
dense_99_807184
identityЂ!conv2d_81/StatefulPartitionedCallЂ!conv2d_82/StatefulPartitionedCallЂ!conv2d_83/StatefulPartitionedCallЂ!conv2d_84/StatefulPartitionedCallЂ dense_96/StatefulPartitionedCallЂ dense_97/StatefulPartitionedCallЂ dense_98/StatefulPartitionedCallЂ dense_99/StatefulPartitionedCallЂ"dropout_20/StatefulPartitionedCallЂ3random_rotation_22/stateful_uniform/StatefulUniformЂ,spatial_dropout2d_22/StatefulPartitionedCallп
8random_flip_22/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:џџџџџџџџџ2:
8random_flip_22/random_flip_left_right/control_dependencyЫ
+random_flip_22/random_flip_left_right/ShapeShapeArandom_flip_22/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2-
+random_flip_22/random_flip_left_right/ShapeР
9random_flip_22/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9random_flip_22/random_flip_left_right/strided_slice/stackФ
;random_flip_22/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip_22/random_flip_left_right/strided_slice/stack_1Ф
;random_flip_22/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip_22/random_flip_left_right/strided_slice/stack_2Ц
3random_flip_22/random_flip_left_right/strided_sliceStridedSlice4random_flip_22/random_flip_left_right/Shape:output:0Brandom_flip_22/random_flip_left_right/strided_slice/stack:output:0Drandom_flip_22/random_flip_left_right/strided_slice/stack_1:output:0Drandom_flip_22/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3random_flip_22/random_flip_left_right/strided_sliceь
:random_flip_22/random_flip_left_right/random_uniform/shapePack<random_flip_22/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2<
:random_flip_22/random_flip_left_right/random_uniform/shapeЙ
8random_flip_22/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2:
8random_flip_22/random_flip_left_right/random_uniform/minЙ
8random_flip_22/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8random_flip_22/random_flip_left_right/random_uniform/max
Brandom_flip_22/random_flip_left_right/random_uniform/RandomUniformRandomUniformCrandom_flip_22/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02D
Brandom_flip_22/random_flip_left_right/random_uniform/RandomUniformЙ
8random_flip_22/random_flip_left_right/random_uniform/MulMulKrandom_flip_22/random_flip_left_right/random_uniform/RandomUniform:output:0Arandom_flip_22/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2:
8random_flip_22/random_flip_left_right/random_uniform/MulА
5random_flip_22/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5random_flip_22/random_flip_left_right/Reshape/shape/1А
5random_flip_22/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :27
5random_flip_22/random_flip_left_right/Reshape/shape/2А
5random_flip_22/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :27
5random_flip_22/random_flip_left_right/Reshape/shape/3
3random_flip_22/random_flip_left_right/Reshape/shapePack<random_flip_22/random_flip_left_right/strided_slice:output:0>random_flip_22/random_flip_left_right/Reshape/shape/1:output:0>random_flip_22/random_flip_left_right/Reshape/shape/2:output:0>random_flip_22/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:25
3random_flip_22/random_flip_left_right/Reshape/shape
-random_flip_22/random_flip_left_right/ReshapeReshape<random_flip_22/random_flip_left_right/random_uniform/Mul:z:0<random_flip_22/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2/
-random_flip_22/random_flip_left_right/Reshapeе
+random_flip_22/random_flip_left_right/RoundRound6random_flip_22/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2-
+random_flip_22/random_flip_left_right/RoundЖ
4random_flip_22/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:26
4random_flip_22/random_flip_left_right/ReverseV2/axis­
/random_flip_22/random_flip_left_right/ReverseV2	ReverseV2Arandom_flip_22/random_flip_left_right/control_dependency:output:0=random_flip_22/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ21
/random_flip_22/random_flip_left_right/ReverseV2
)random_flip_22/random_flip_left_right/mulMul/random_flip_22/random_flip_left_right/Round:y:08random_flip_22/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2+
)random_flip_22/random_flip_left_right/mul
+random_flip_22/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+random_flip_22/random_flip_left_right/sub/xў
)random_flip_22/random_flip_left_right/subSub4random_flip_22/random_flip_left_right/sub/x:output:0/random_flip_22/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2+
)random_flip_22/random_flip_left_right/sub
+random_flip_22/random_flip_left_right/mul_1Mul-random_flip_22/random_flip_left_right/sub:z:0Arandom_flip_22/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2-
+random_flip_22/random_flip_left_right/mul_1ћ
)random_flip_22/random_flip_left_right/addAddV2-random_flip_22/random_flip_left_right/mul:z:0/random_flip_22/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2+
)random_flip_22/random_flip_left_right/addЃ
5random_flip_22/random_flip_up_down/control_dependencyIdentity-random_flip_22/random_flip_left_right/add:z:0*
T0*<
_class2
0.loc:@random_flip_22/random_flip_left_right/add*1
_output_shapes
:џџџџџџџџџ27
5random_flip_22/random_flip_up_down/control_dependencyТ
(random_flip_22/random_flip_up_down/ShapeShape>random_flip_22/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:2*
(random_flip_22/random_flip_up_down/ShapeК
6random_flip_22/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_flip_22/random_flip_up_down/strided_slice/stackО
8random_flip_22/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip_22/random_flip_up_down/strided_slice/stack_1О
8random_flip_22/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip_22/random_flip_up_down/strided_slice/stack_2Д
0random_flip_22/random_flip_up_down/strided_sliceStridedSlice1random_flip_22/random_flip_up_down/Shape:output:0?random_flip_22/random_flip_up_down/strided_slice/stack:output:0Arandom_flip_22/random_flip_up_down/strided_slice/stack_1:output:0Arandom_flip_22/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_flip_22/random_flip_up_down/strided_sliceу
7random_flip_22/random_flip_up_down/random_uniform/shapePack9random_flip_22/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:29
7random_flip_22/random_flip_up_down/random_uniform/shapeГ
5random_flip_22/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5random_flip_22/random_flip_up_down/random_uniform/minГ
5random_flip_22/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5random_flip_22/random_flip_up_down/random_uniform/max
?random_flip_22/random_flip_up_down/random_uniform/RandomUniformRandomUniform@random_flip_22/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02A
?random_flip_22/random_flip_up_down/random_uniform/RandomUniform­
5random_flip_22/random_flip_up_down/random_uniform/MulMulHrandom_flip_22/random_flip_up_down/random_uniform/RandomUniform:output:0>random_flip_22/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ27
5random_flip_22/random_flip_up_down/random_uniform/MulЊ
2random_flip_22/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip_22/random_flip_up_down/Reshape/shape/1Њ
2random_flip_22/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip_22/random_flip_up_down/Reshape/shape/2Њ
2random_flip_22/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip_22/random_flip_up_down/Reshape/shape/3
0random_flip_22/random_flip_up_down/Reshape/shapePack9random_flip_22/random_flip_up_down/strided_slice:output:0;random_flip_22/random_flip_up_down/Reshape/shape/1:output:0;random_flip_22/random_flip_up_down/Reshape/shape/2:output:0;random_flip_22/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:22
0random_flip_22/random_flip_up_down/Reshape/shape
*random_flip_22/random_flip_up_down/ReshapeReshape9random_flip_22/random_flip_up_down/random_uniform/Mul:z:09random_flip_22/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2,
*random_flip_22/random_flip_up_down/ReshapeЬ
(random_flip_22/random_flip_up_down/RoundRound3random_flip_22/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
(random_flip_22/random_flip_up_down/RoundА
1random_flip_22/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:23
1random_flip_22/random_flip_up_down/ReverseV2/axisЁ
,random_flip_22/random_flip_up_down/ReverseV2	ReverseV2>random_flip_22/random_flip_up_down/control_dependency:output:0:random_flip_22/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2.
,random_flip_22/random_flip_up_down/ReverseV2ј
&random_flip_22/random_flip_up_down/mulMul,random_flip_22/random_flip_up_down/Round:y:05random_flip_22/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2(
&random_flip_22/random_flip_up_down/mul
(random_flip_22/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(random_flip_22/random_flip_up_down/sub/xђ
&random_flip_22/random_flip_up_down/subSub1random_flip_22/random_flip_up_down/sub/x:output:0,random_flip_22/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2(
&random_flip_22/random_flip_up_down/sub
(random_flip_22/random_flip_up_down/mul_1Mul*random_flip_22/random_flip_up_down/sub:z:0>random_flip_22/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2*
(random_flip_22/random_flip_up_down/mul_1я
&random_flip_22/random_flip_up_down/addAddV2*random_flip_22/random_flip_up_down/mul:z:0,random_flip_22/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2(
&random_flip_22/random_flip_up_down/add
random_rotation_22/ShapeShape*random_flip_22/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2
random_rotation_22/Shape
&random_rotation_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_rotation_22/strided_slice/stack
(random_rotation_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice/stack_1
(random_rotation_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice/stack_2д
 random_rotation_22/strided_sliceStridedSlice!random_rotation_22/Shape:output:0/random_rotation_22/strided_slice/stack:output:01random_rotation_22/strided_slice/stack_1:output:01random_rotation_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 random_rotation_22/strided_slice
(random_rotation_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice_1/stackЂ
*random_rotation_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_1/stack_1Ђ
*random_rotation_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_1/stack_2о
"random_rotation_22/strided_slice_1StridedSlice!random_rotation_22/Shape:output:01random_rotation_22/strided_slice_1/stack:output:03random_rotation_22/strided_slice_1/stack_1:output:03random_rotation_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_rotation_22/strided_slice_1
random_rotation_22/CastCast+random_rotation_22/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_22/Cast
(random_rotation_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice_2/stackЂ
*random_rotation_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_2/stack_1Ђ
*random_rotation_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_2/stack_2о
"random_rotation_22/strided_slice_2StridedSlice!random_rotation_22/Shape:output:01random_rotation_22/strided_slice_2/stack:output:03random_rotation_22/strided_slice_2/stack_1:output:03random_rotation_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_rotation_22/strided_slice_2
random_rotation_22/Cast_1Cast+random_rotation_22/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_22/Cast_1З
)random_rotation_22/stateful_uniform/shapePack)random_rotation_22/strided_slice:output:0*
N*
T0*
_output_shapes
:2+
)random_rotation_22/stateful_uniform/shape
'random_rotation_22/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|й Р2)
'random_rotation_22/stateful_uniform/min
'random_rotation_22/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|й @2)
'random_rotation_22/stateful_uniform/maxР
=random_rotation_22/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2?
=random_rotation_22/stateful_uniform/StatefulUniform/algorithmя
3random_rotation_22/stateful_uniform/StatefulUniformStatefulUniform<random_rotation_22_stateful_uniform_statefuluniform_resourceFrandom_rotation_22/stateful_uniform/StatefulUniform/algorithm:output:02random_rotation_22/stateful_uniform/shape:output:0*#
_output_shapes
:џџџџџџџџџ*
shape_dtype025
3random_rotation_22/stateful_uniform/StatefulUniformо
'random_rotation_22/stateful_uniform/subSub0random_rotation_22/stateful_uniform/max:output:00random_rotation_22/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2)
'random_rotation_22/stateful_uniform/subђ
'random_rotation_22/stateful_uniform/mulMul<random_rotation_22/stateful_uniform/StatefulUniform:output:0+random_rotation_22/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_22/stateful_uniform/mulо
#random_rotation_22/stateful_uniformAdd+random_rotation_22/stateful_uniform/mul:z:00random_rotation_22/stateful_uniform/min:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2%
#random_rotation_22/stateful_uniform
(random_rotation_22/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(random_rotation_22/rotation_matrix/sub/yЪ
&random_rotation_22/rotation_matrix/subSubrandom_rotation_22/Cast_1:y:01random_rotation_22/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2(
&random_rotation_22/rotation_matrix/subЎ
&random_rotation_22/rotation_matrix/CosCos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/Cos
*random_rotation_22/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_1/yа
(random_rotation_22/rotation_matrix/sub_1Subrandom_rotation_22/Cast_1:y:03random_rotation_22/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_1п
&random_rotation_22/rotation_matrix/mulMul*random_rotation_22/rotation_matrix/Cos:y:0,random_rotation_22/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/mulЎ
&random_rotation_22/rotation_matrix/SinSin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/Sin
*random_rotation_22/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_2/yЮ
(random_rotation_22/rotation_matrix/sub_2Subrandom_rotation_22/Cast:y:03random_rotation_22/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_2у
(random_rotation_22/rotation_matrix/mul_1Mul*random_rotation_22/rotation_matrix/Sin:y:0,random_rotation_22/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/mul_1у
(random_rotation_22/rotation_matrix/sub_3Sub*random_rotation_22/rotation_matrix/mul:z:0,random_rotation_22/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/sub_3у
(random_rotation_22/rotation_matrix/sub_4Sub*random_rotation_22/rotation_matrix/sub:z:0,random_rotation_22/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/sub_4Ё
,random_rotation_22/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,random_rotation_22/rotation_matrix/truediv/yі
*random_rotation_22/rotation_matrix/truedivRealDiv,random_rotation_22/rotation_matrix/sub_4:z:05random_rotation_22/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2,
*random_rotation_22/rotation_matrix/truediv
*random_rotation_22/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_5/yЮ
(random_rotation_22/rotation_matrix/sub_5Subrandom_rotation_22/Cast:y:03random_rotation_22/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_5В
(random_rotation_22/rotation_matrix/Sin_1Sin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Sin_1
*random_rotation_22/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_6/yа
(random_rotation_22/rotation_matrix/sub_6Subrandom_rotation_22/Cast_1:y:03random_rotation_22/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_6х
(random_rotation_22/rotation_matrix/mul_2Mul,random_rotation_22/rotation_matrix/Sin_1:y:0,random_rotation_22/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/mul_2В
(random_rotation_22/rotation_matrix/Cos_1Cos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Cos_1
*random_rotation_22/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_7/yЮ
(random_rotation_22/rotation_matrix/sub_7Subrandom_rotation_22/Cast:y:03random_rotation_22/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_7х
(random_rotation_22/rotation_matrix/mul_3Mul,random_rotation_22/rotation_matrix/Cos_1:y:0,random_rotation_22/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/mul_3у
&random_rotation_22/rotation_matrix/addAddV2,random_rotation_22/rotation_matrix/mul_2:z:0,random_rotation_22/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/addу
(random_rotation_22/rotation_matrix/sub_8Sub,random_rotation_22/rotation_matrix/sub_5:z:0*random_rotation_22/rotation_matrix/add:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/sub_8Ѕ
.random_rotation_22/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.random_rotation_22/rotation_matrix/truediv_1/yќ
,random_rotation_22/rotation_matrix/truediv_1RealDiv,random_rotation_22/rotation_matrix/sub_8:z:07random_rotation_22/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2.
,random_rotation_22/rotation_matrix/truediv_1Ћ
(random_rotation_22/rotation_matrix/ShapeShape'random_rotation_22/stateful_uniform:z:0*
T0*
_output_shapes
:2*
(random_rotation_22/rotation_matrix/ShapeК
6random_rotation_22/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_rotation_22/rotation_matrix/strided_slice/stackО
8random_rotation_22/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_22/rotation_matrix/strided_slice/stack_1О
8random_rotation_22/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_22/rotation_matrix/strided_slice/stack_2Д
0random_rotation_22/rotation_matrix/strided_sliceStridedSlice1random_rotation_22/rotation_matrix/Shape:output:0?random_rotation_22/rotation_matrix/strided_slice/stack:output:0Arandom_rotation_22/rotation_matrix/strided_slice/stack_1:output:0Arandom_rotation_22/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_rotation_22/rotation_matrix/strided_sliceВ
(random_rotation_22/rotation_matrix/Cos_2Cos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Cos_2Х
8random_rotation_22/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_1/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_1/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_1/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_1StridedSlice,random_rotation_22/rotation_matrix/Cos_2:y:0Arandom_rotation_22/rotation_matrix/strided_slice_1/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_1/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_1В
(random_rotation_22/rotation_matrix/Sin_2Sin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Sin_2Х
8random_rotation_22/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_2/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_2/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_2/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_2StridedSlice,random_rotation_22/rotation_matrix/Sin_2:y:0Arandom_rotation_22/rotation_matrix/strided_slice_2/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_2/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_2Ц
&random_rotation_22/rotation_matrix/NegNeg;random_rotation_22/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/NegХ
8random_rotation_22/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_3/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_3/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_3/stack_2ы
2random_rotation_22/rotation_matrix/strided_slice_3StridedSlice.random_rotation_22/rotation_matrix/truediv:z:0Arandom_rotation_22/rotation_matrix/strided_slice_3/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_3/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_3В
(random_rotation_22/rotation_matrix/Sin_3Sin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Sin_3Х
8random_rotation_22/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_4/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_4/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_4/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_4StridedSlice,random_rotation_22/rotation_matrix/Sin_3:y:0Arandom_rotation_22/rotation_matrix/strided_slice_4/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_4/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_4В
(random_rotation_22/rotation_matrix/Cos_3Cos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Cos_3Х
8random_rotation_22/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_5/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_5/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_5/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_5StridedSlice,random_rotation_22/rotation_matrix/Cos_3:y:0Arandom_rotation_22/rotation_matrix/strided_slice_5/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_5/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_5Х
8random_rotation_22/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_6/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_6/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_6/stack_2э
2random_rotation_22/rotation_matrix/strided_slice_6StridedSlice0random_rotation_22/rotation_matrix/truediv_1:z:0Arandom_rotation_22/rotation_matrix/strided_slice_6/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_6/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_6Ђ
.random_rotation_22/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation_22/rotation_matrix/zeros/mul/yј
,random_rotation_22/rotation_matrix/zeros/mulMul9random_rotation_22/rotation_matrix/strided_slice:output:07random_rotation_22/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2.
,random_rotation_22/rotation_matrix/zeros/mulЅ
/random_rotation_22/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш21
/random_rotation_22/rotation_matrix/zeros/Less/yѓ
-random_rotation_22/rotation_matrix/zeros/LessLess0random_rotation_22/rotation_matrix/zeros/mul:z:08random_rotation_22/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2/
-random_rotation_22/rotation_matrix/zeros/LessЈ
1random_rotation_22/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :23
1random_rotation_22/rotation_matrix/zeros/packed/1
/random_rotation_22/rotation_matrix/zeros/packedPack9random_rotation_22/rotation_matrix/strided_slice:output:0:random_rotation_22/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:21
/random_rotation_22/rotation_matrix/zeros/packedЅ
.random_rotation_22/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.random_rotation_22/rotation_matrix/zeros/Const
(random_rotation_22/rotation_matrix/zerosFill8random_rotation_22/rotation_matrix/zeros/packed:output:07random_rotation_22/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/zerosЂ
.random_rotation_22/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation_22/rotation_matrix/concat/axisц
)random_rotation_22/rotation_matrix/concatConcatV2;random_rotation_22/rotation_matrix/strided_slice_1:output:0*random_rotation_22/rotation_matrix/Neg:y:0;random_rotation_22/rotation_matrix/strided_slice_3:output:0;random_rotation_22/rotation_matrix/strided_slice_4:output:0;random_rotation_22/rotation_matrix/strided_slice_5:output:0;random_rotation_22/rotation_matrix/strided_slice_6:output:01random_rotation_22/rotation_matrix/zeros:output:07random_rotation_22/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2+
)random_rotation_22/rotation_matrix/concatЂ
"random_rotation_22/transform/ShapeShape*random_flip_22/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2$
"random_rotation_22/transform/ShapeЎ
0random_rotation_22/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0random_rotation_22/transform/strided_slice/stackВ
2random_rotation_22/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2random_rotation_22/transform/strided_slice/stack_1В
2random_rotation_22/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2random_rotation_22/transform/strided_slice/stack_2ќ
*random_rotation_22/transform/strided_sliceStridedSlice+random_rotation_22/transform/Shape:output:09random_rotation_22/transform/strided_slice/stack:output:0;random_rotation_22/transform/strided_slice/stack_1:output:0;random_rotation_22/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*random_rotation_22/transform/strided_slice
'random_rotation_22/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_rotation_22/transform/fill_valueЬ
7random_rotation_22/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3*random_flip_22/random_flip_up_down/add:z:02random_rotation_22/rotation_matrix/concat:output:03random_rotation_22/transform/strided_slice:output:00random_rotation_22/transform/fill_value:output:0*1
_output_shapes
:џџџџџџџџџ*
dtype0*
	fill_mode	NEAREST*
interpolation
BILINEAR29
7random_rotation_22/transform/ImageProjectiveTransformV3ь
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCallLrandom_rotation_22/transform/ImageProjectiveTransformV3:transformed_images:0conv2d_81_807140conv2d_81_807142*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџўў *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_81_layer_call_and_return_conditional_losses_8066372#
!conv2d_81/StatefulPartitionedCall
 max_pooling2d_81/PartitionedCallPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_8063432"
 max_pooling2d_81/PartitionedCallЩ
!conv2d_82/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_81/PartitionedCall:output:0conv2d_82_807146conv2d_82_807148*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ§§ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_82_layer_call_and_return_conditional_losses_8066702#
!conv2d_82/StatefulPartitionedCall
 max_pooling2d_82/PartitionedCallPartitionedCall*conv2d_82/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ~~ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_8063552"
 max_pooling2d_82/PartitionedCallЧ
!conv2d_83/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_82/PartitionedCall:output:0conv2d_83_807152conv2d_83_807154*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ||@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_83_layer_call_and_return_conditional_losses_8067032#
!conv2d_83/StatefulPartitionedCall
 max_pooling2d_83/PartitionedCallPartitionedCall*conv2d_83/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ>>@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_8063672"
 max_pooling2d_83/PartitionedCallЧ
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_83/PartitionedCall:output:0conv2d_84_807158conv2d_84_807160*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_84_layer_call_and_return_conditional_losses_8067362#
!conv2d_84/StatefulPartitionedCall
 max_pooling2d_84/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_8063792"
 max_pooling2d_84/PartitionedCallО
,spatial_dropout2d_22/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_84/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_8067752.
,spatial_dropout2d_22/StatefulPartitionedCallП
+global_average_pooling2d_22/PartitionedCallPartitionedCall5spatial_dropout2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_22_layer_call_and_return_conditional_losses_8064602-
+global_average_pooling2d_22/PartitionedCallЦ
 dense_96/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_22/PartitionedCall:output:0dense_96_807166dense_96_807168*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_8068042"
 dense_96/StatefulPartitionedCallЛ
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_807171dense_97_807173*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_8068312"
 dense_97/StatefulPartitionedCallК
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_807176dense_98_807178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_98_layer_call_and_return_conditional_losses_8068582"
 dense_98/StatefulPartitionedCallЧ
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall)dense_98/StatefulPartitionedCall:output:0-^spatial_dropout2d_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_20_layer_call_and_return_conditional_losses_8068862$
"dropout_20/StatefulPartitionedCallМ
 dense_99/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0dense_99_807182dense_99_807184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_8069152"
 dense_99/StatefulPartitionedCallЃ
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0"^conv2d_81/StatefulPartitionedCall"^conv2d_82/StatefulPartitionedCall"^conv2d_83/StatefulPartitionedCall"^conv2d_84/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall4^random_rotation_22/stateful_uniform/StatefulUniform-^spatial_dropout2d_22/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:џџџџџџџџџ:::::::::::::::::2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2F
!conv2d_82/StatefulPartitionedCall!conv2d_82/StatefulPartitionedCall2F
!conv2d_83/StatefulPartitionedCall!conv2d_83/StatefulPartitionedCall2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2j
3random_rotation_22/stateful_uniform/StatefulUniform3random_rotation_22/stateful_uniform/StatefulUniform2\
,spatial_dropout2d_22/StatefulPartitionedCall,spatial_dropout2d_22/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р
n
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_806450

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
щ

!__inference__wrapped_model_806337
random_flip_22_input:
6sequential_22_conv2d_81_conv2d_readvariableop_resource;
7sequential_22_conv2d_81_biasadd_readvariableop_resource:
6sequential_22_conv2d_82_conv2d_readvariableop_resource;
7sequential_22_conv2d_82_biasadd_readvariableop_resource:
6sequential_22_conv2d_83_conv2d_readvariableop_resource;
7sequential_22_conv2d_83_biasadd_readvariableop_resource:
6sequential_22_conv2d_84_conv2d_readvariableop_resource;
7sequential_22_conv2d_84_biasadd_readvariableop_resource9
5sequential_22_dense_96_matmul_readvariableop_resource:
6sequential_22_dense_96_biasadd_readvariableop_resource9
5sequential_22_dense_97_matmul_readvariableop_resource:
6sequential_22_dense_97_biasadd_readvariableop_resource9
5sequential_22_dense_98_matmul_readvariableop_resource:
6sequential_22_dense_98_biasadd_readvariableop_resource9
5sequential_22_dense_99_matmul_readvariableop_resource:
6sequential_22_dense_99_biasadd_readvariableop_resource
identityЂ.sequential_22/conv2d_81/BiasAdd/ReadVariableOpЂ-sequential_22/conv2d_81/Conv2D/ReadVariableOpЂ.sequential_22/conv2d_82/BiasAdd/ReadVariableOpЂ-sequential_22/conv2d_82/Conv2D/ReadVariableOpЂ.sequential_22/conv2d_83/BiasAdd/ReadVariableOpЂ-sequential_22/conv2d_83/Conv2D/ReadVariableOpЂ.sequential_22/conv2d_84/BiasAdd/ReadVariableOpЂ-sequential_22/conv2d_84/Conv2D/ReadVariableOpЂ-sequential_22/dense_96/BiasAdd/ReadVariableOpЂ,sequential_22/dense_96/MatMul/ReadVariableOpЂ-sequential_22/dense_97/BiasAdd/ReadVariableOpЂ,sequential_22/dense_97/MatMul/ReadVariableOpЂ-sequential_22/dense_98/BiasAdd/ReadVariableOpЂ,sequential_22/dense_98/MatMul/ReadVariableOpЂ-sequential_22/dense_99/BiasAdd/ReadVariableOpЂ,sequential_22/dense_99/MatMul/ReadVariableOpн
-sequential_22/conv2d_81/Conv2D/ReadVariableOpReadVariableOp6sequential_22_conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_22/conv2d_81/Conv2D/ReadVariableOpќ
sequential_22/conv2d_81/Conv2DConv2Drandom_flip_22_input5sequential_22/conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў *
paddingVALID*
strides
2 
sequential_22/conv2d_81/Conv2Dд
.sequential_22/conv2d_81/BiasAdd/ReadVariableOpReadVariableOp7sequential_22_conv2d_81_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_22/conv2d_81/BiasAdd/ReadVariableOpъ
sequential_22/conv2d_81/BiasAddBiasAdd'sequential_22/conv2d_81/Conv2D:output:06sequential_22/conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2!
sequential_22/conv2d_81/BiasAddГ
sequential_22/conv2d_81/SigmoidSigmoid(sequential_22/conv2d_81/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2!
sequential_22/conv2d_81/SigmoidЬ
sequential_22/conv2d_81/mulMul(sequential_22/conv2d_81/BiasAdd:output:0#sequential_22/conv2d_81/Sigmoid:y:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2
sequential_22/conv2d_81/mul­
 sequential_22/conv2d_81/IdentityIdentitysequential_22/conv2d_81/mul:z:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2"
 sequential_22/conv2d_81/IdentityЉ
!sequential_22/conv2d_81/IdentityN	IdentityNsequential_22/conv2d_81/mul:z:0(sequential_22/conv2d_81/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-806258*N
_output_shapes<
::џџџџџџџџџўў :џџџџџџџџџўў 2#
!sequential_22/conv2d_81/IdentityNі
&sequential_22/max_pooling2d_81/MaxPoolMaxPool*sequential_22/conv2d_81/IdentityN:output:0*1
_output_shapes
:џџџџџџџџџџџ *
ksize
*
paddingVALID*
strides
2(
&sequential_22/max_pooling2d_81/MaxPoolн
-sequential_22/conv2d_82/Conv2D/ReadVariableOpReadVariableOp6sequential_22_conv2d_82_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02/
-sequential_22/conv2d_82/Conv2D/ReadVariableOp
sequential_22/conv2d_82/Conv2DConv2D/sequential_22/max_pooling2d_81/MaxPool:output:05sequential_22/conv2d_82/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ *
paddingVALID*
strides
2 
sequential_22/conv2d_82/Conv2Dд
.sequential_22/conv2d_82/BiasAdd/ReadVariableOpReadVariableOp7sequential_22_conv2d_82_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_22/conv2d_82/BiasAdd/ReadVariableOpъ
sequential_22/conv2d_82/BiasAddBiasAdd'sequential_22/conv2d_82/Conv2D:output:06sequential_22/conv2d_82/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2!
sequential_22/conv2d_82/BiasAddГ
sequential_22/conv2d_82/SigmoidSigmoid(sequential_22/conv2d_82/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2!
sequential_22/conv2d_82/SigmoidЬ
sequential_22/conv2d_82/mulMul(sequential_22/conv2d_82/BiasAdd:output:0#sequential_22/conv2d_82/Sigmoid:y:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2
sequential_22/conv2d_82/mul­
 sequential_22/conv2d_82/IdentityIdentitysequential_22/conv2d_82/mul:z:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2"
 sequential_22/conv2d_82/IdentityЉ
!sequential_22/conv2d_82/IdentityN	IdentityNsequential_22/conv2d_82/mul:z:0(sequential_22/conv2d_82/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-806271*N
_output_shapes<
::џџџџџџџџџ§§ :џџџџџџџџџ§§ 2#
!sequential_22/conv2d_82/IdentityNє
&sequential_22/max_pooling2d_82/MaxPoolMaxPool*sequential_22/conv2d_82/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ~~ *
ksize
*
paddingVALID*
strides
2(
&sequential_22/max_pooling2d_82/MaxPoolн
-sequential_22/conv2d_83/Conv2D/ReadVariableOpReadVariableOp6sequential_22_conv2d_83_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_22/conv2d_83/Conv2D/ReadVariableOp
sequential_22/conv2d_83/Conv2DConv2D/sequential_22/max_pooling2d_82/MaxPool:output:05sequential_22/conv2d_83/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ||@*
paddingVALID*
strides
2 
sequential_22/conv2d_83/Conv2Dд
.sequential_22/conv2d_83/BiasAdd/ReadVariableOpReadVariableOp7sequential_22_conv2d_83_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_22/conv2d_83/BiasAdd/ReadVariableOpш
sequential_22/conv2d_83/BiasAddBiasAdd'sequential_22/conv2d_83/Conv2D:output:06sequential_22/conv2d_83/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2!
sequential_22/conv2d_83/BiasAddБ
sequential_22/conv2d_83/SigmoidSigmoid(sequential_22/conv2d_83/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2!
sequential_22/conv2d_83/SigmoidЪ
sequential_22/conv2d_83/mulMul(sequential_22/conv2d_83/BiasAdd:output:0#sequential_22/conv2d_83/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2
sequential_22/conv2d_83/mulЋ
 sequential_22/conv2d_83/IdentityIdentitysequential_22/conv2d_83/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2"
 sequential_22/conv2d_83/IdentityЅ
!sequential_22/conv2d_83/IdentityN	IdentityNsequential_22/conv2d_83/mul:z:0(sequential_22/conv2d_83/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-806284*J
_output_shapes8
6:џџџџџџџџџ||@:џџџџџџџџџ||@2#
!sequential_22/conv2d_83/IdentityNє
&sequential_22/max_pooling2d_83/MaxPoolMaxPool*sequential_22/conv2d_83/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ>>@*
ksize
*
paddingVALID*
strides
2(
&sequential_22/max_pooling2d_83/MaxPoolн
-sequential_22/conv2d_84/Conv2D/ReadVariableOpReadVariableOp6sequential_22_conv2d_84_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_22/conv2d_84/Conv2D/ReadVariableOp
sequential_22/conv2d_84/Conv2DConv2D/sequential_22/max_pooling2d_83/MaxPool:output:05sequential_22/conv2d_84/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@*
paddingVALID*
strides
2 
sequential_22/conv2d_84/Conv2Dд
.sequential_22/conv2d_84/BiasAdd/ReadVariableOpReadVariableOp7sequential_22_conv2d_84_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_22/conv2d_84/BiasAdd/ReadVariableOpш
sequential_22/conv2d_84/BiasAddBiasAdd'sequential_22/conv2d_84/Conv2D:output:06sequential_22/conv2d_84/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2!
sequential_22/conv2d_84/BiasAddБ
sequential_22/conv2d_84/SigmoidSigmoid(sequential_22/conv2d_84/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2!
sequential_22/conv2d_84/SigmoidЪ
sequential_22/conv2d_84/mulMul(sequential_22/conv2d_84/BiasAdd:output:0#sequential_22/conv2d_84/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
sequential_22/conv2d_84/mulЋ
 sequential_22/conv2d_84/IdentityIdentitysequential_22/conv2d_84/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2"
 sequential_22/conv2d_84/IdentityЅ
!sequential_22/conv2d_84/IdentityN	IdentityNsequential_22/conv2d_84/mul:z:0(sequential_22/conv2d_84/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-806297*J
_output_shapes8
6:џџџџџџџџџ<<@:џџџџџџџџџ<<@2#
!sequential_22/conv2d_84/IdentityNє
&sequential_22/max_pooling2d_84/MaxPoolMaxPool*sequential_22/conv2d_84/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2(
&sequential_22/max_pooling2d_84/MaxPoolб
+sequential_22/spatial_dropout2d_22/IdentityIdentity/sequential_22/max_pooling2d_84/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2-
+sequential_22/spatial_dropout2d_22/Identityе
@sequential_22/global_average_pooling2d_22/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_22/global_average_pooling2d_22/Mean/reduction_indices
.sequential_22/global_average_pooling2d_22/MeanMean4sequential_22/spatial_dropout2d_22/Identity:output:0Isequential_22/global_average_pooling2d_22/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@20
.sequential_22/global_average_pooling2d_22/Meanг
,sequential_22/dense_96/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_96_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,sequential_22/dense_96/MatMul/ReadVariableOpъ
sequential_22/dense_96/MatMulMatMul7sequential_22/global_average_pooling2d_22/Mean:output:04sequential_22/dense_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential_22/dense_96/MatMulв
-sequential_22/dense_96/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_96_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_22/dense_96/BiasAdd/ReadVariableOpо
sequential_22/dense_96/BiasAddBiasAdd'sequential_22/dense_96/MatMul:product:05sequential_22/dense_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
sequential_22/dense_96/BiasAdd
sequential_22/dense_96/ReluRelu'sequential_22/dense_96/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential_22/dense_96/Reluд
,sequential_22/dense_97/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_97_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,sequential_22/dense_97/MatMul/ReadVariableOpм
sequential_22/dense_97/MatMulMatMul)sequential_22/dense_96/Relu:activations:04sequential_22/dense_97/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential_22/dense_97/MatMulв
-sequential_22/dense_97/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_97_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_22/dense_97/BiasAdd/ReadVariableOpо
sequential_22/dense_97/BiasAddBiasAdd'sequential_22/dense_97/MatMul:product:05sequential_22/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
sequential_22/dense_97/BiasAdd
sequential_22/dense_97/ReluRelu'sequential_22/dense_97/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential_22/dense_97/Reluг
,sequential_22/dense_98/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_98_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,sequential_22/dense_98/MatMul/ReadVariableOpл
sequential_22/dense_98/MatMulMatMul)sequential_22/dense_97/Relu:activations:04sequential_22/dense_98/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential_22/dense_98/MatMulб
-sequential_22/dense_98/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_22/dense_98/BiasAdd/ReadVariableOpн
sequential_22/dense_98/BiasAddBiasAdd'sequential_22/dense_98/MatMul:product:05sequential_22/dense_98/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2 
sequential_22/dense_98/BiasAdd
sequential_22/dense_98/ReluRelu'sequential_22/dense_98/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential_22/dense_98/ReluЏ
!sequential_22/dropout_20/IdentityIdentity)sequential_22/dense_98/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2#
!sequential_22/dropout_20/Identityв
,sequential_22/dense_99/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_99_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,sequential_22/dense_99/MatMul/ReadVariableOpм
sequential_22/dense_99/MatMulMatMul*sequential_22/dropout_20/Identity:output:04sequential_22/dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_22/dense_99/MatMulб
-sequential_22/dense_99/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_99_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_22/dense_99/BiasAdd/ReadVariableOpн
sequential_22/dense_99/BiasAddBiasAdd'sequential_22/dense_99/MatMul:product:05sequential_22/dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_22/dense_99/BiasAddІ
sequential_22/dense_99/SoftmaxSoftmax'sequential_22/dense_99/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_22/dense_99/Softmaxќ
IdentityIdentity(sequential_22/dense_99/Softmax:softmax:0/^sequential_22/conv2d_81/BiasAdd/ReadVariableOp.^sequential_22/conv2d_81/Conv2D/ReadVariableOp/^sequential_22/conv2d_82/BiasAdd/ReadVariableOp.^sequential_22/conv2d_82/Conv2D/ReadVariableOp/^sequential_22/conv2d_83/BiasAdd/ReadVariableOp.^sequential_22/conv2d_83/Conv2D/ReadVariableOp/^sequential_22/conv2d_84/BiasAdd/ReadVariableOp.^sequential_22/conv2d_84/Conv2D/ReadVariableOp.^sequential_22/dense_96/BiasAdd/ReadVariableOp-^sequential_22/dense_96/MatMul/ReadVariableOp.^sequential_22/dense_97/BiasAdd/ReadVariableOp-^sequential_22/dense_97/MatMul/ReadVariableOp.^sequential_22/dense_98/BiasAdd/ReadVariableOp-^sequential_22/dense_98/MatMul/ReadVariableOp.^sequential_22/dense_99/BiasAdd/ReadVariableOp-^sequential_22/dense_99/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::2`
.sequential_22/conv2d_81/BiasAdd/ReadVariableOp.sequential_22/conv2d_81/BiasAdd/ReadVariableOp2^
-sequential_22/conv2d_81/Conv2D/ReadVariableOp-sequential_22/conv2d_81/Conv2D/ReadVariableOp2`
.sequential_22/conv2d_82/BiasAdd/ReadVariableOp.sequential_22/conv2d_82/BiasAdd/ReadVariableOp2^
-sequential_22/conv2d_82/Conv2D/ReadVariableOp-sequential_22/conv2d_82/Conv2D/ReadVariableOp2`
.sequential_22/conv2d_83/BiasAdd/ReadVariableOp.sequential_22/conv2d_83/BiasAdd/ReadVariableOp2^
-sequential_22/conv2d_83/Conv2D/ReadVariableOp-sequential_22/conv2d_83/Conv2D/ReadVariableOp2`
.sequential_22/conv2d_84/BiasAdd/ReadVariableOp.sequential_22/conv2d_84/BiasAdd/ReadVariableOp2^
-sequential_22/conv2d_84/Conv2D/ReadVariableOp-sequential_22/conv2d_84/Conv2D/ReadVariableOp2^
-sequential_22/dense_96/BiasAdd/ReadVariableOp-sequential_22/dense_96/BiasAdd/ReadVariableOp2\
,sequential_22/dense_96/MatMul/ReadVariableOp,sequential_22/dense_96/MatMul/ReadVariableOp2^
-sequential_22/dense_97/BiasAdd/ReadVariableOp-sequential_22/dense_97/BiasAdd/ReadVariableOp2\
,sequential_22/dense_97/MatMul/ReadVariableOp,sequential_22/dense_97/MatMul/ReadVariableOp2^
-sequential_22/dense_98/BiasAdd/ReadVariableOp-sequential_22/dense_98/BiasAdd/ReadVariableOp2\
,sequential_22/dense_98/MatMul/ReadVariableOp,sequential_22/dense_98/MatMul/ReadVariableOp2^
-sequential_22/dense_99/BiasAdd/ReadVariableOp-sequential_22/dense_99/BiasAdd/ReadVariableOp2\
,sequential_22/dense_99/MatMul/ReadVariableOp,sequential_22/dense_99/MatMul/ReadVariableOp:g c
1
_output_shapes
:џџџџџџџџџ
.
_user_specified_namerandom_flip_22_input

р
E__inference_conv2d_82_layer_call_and_return_conditional_losses_806670

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpІ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2	
Sigmoidl
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2

IdentityЩ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-806663*N
_output_shapes<
::џџџџџџџџџ§§ :џџџџџџџџџ§§ 2
	IdentityNЅ

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2

Identity_1"!

identity_1Identity_1:output:0*8
_input_shapes'
%:џџџџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџџџ 
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_806343

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш

к
.__inference_sequential_22_layer_call_fn_807788

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identityЂStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_8072782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:џџџџџџџџџ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р
n
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807954

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
УІ
р
I__inference_sequential_22_layer_call_and_return_conditional_losses_807624

inputs@
<random_rotation_22_stateful_uniform_statefuluniform_resource,
(conv2d_81_conv2d_readvariableop_resource-
)conv2d_81_biasadd_readvariableop_resource,
(conv2d_82_conv2d_readvariableop_resource-
)conv2d_82_biasadd_readvariableop_resource,
(conv2d_83_conv2d_readvariableop_resource-
)conv2d_83_biasadd_readvariableop_resource,
(conv2d_84_conv2d_readvariableop_resource-
)conv2d_84_biasadd_readvariableop_resource+
'dense_96_matmul_readvariableop_resource,
(dense_96_biasadd_readvariableop_resource+
'dense_97_matmul_readvariableop_resource,
(dense_97_biasadd_readvariableop_resource+
'dense_98_matmul_readvariableop_resource,
(dense_98_biasadd_readvariableop_resource+
'dense_99_matmul_readvariableop_resource,
(dense_99_biasadd_readvariableop_resource
identityЂ conv2d_81/BiasAdd/ReadVariableOpЂconv2d_81/Conv2D/ReadVariableOpЂ conv2d_82/BiasAdd/ReadVariableOpЂconv2d_82/Conv2D/ReadVariableOpЂ conv2d_83/BiasAdd/ReadVariableOpЂconv2d_83/Conv2D/ReadVariableOpЂ conv2d_84/BiasAdd/ReadVariableOpЂconv2d_84/Conv2D/ReadVariableOpЂdense_96/BiasAdd/ReadVariableOpЂdense_96/MatMul/ReadVariableOpЂdense_97/BiasAdd/ReadVariableOpЂdense_97/MatMul/ReadVariableOpЂdense_98/BiasAdd/ReadVariableOpЂdense_98/MatMul/ReadVariableOpЂdense_99/BiasAdd/ReadVariableOpЂdense_99/MatMul/ReadVariableOpЂ3random_rotation_22/stateful_uniform/StatefulUniformп
8random_flip_22/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:џџџџџџџџџ2:
8random_flip_22/random_flip_left_right/control_dependencyЫ
+random_flip_22/random_flip_left_right/ShapeShapeArandom_flip_22/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2-
+random_flip_22/random_flip_left_right/ShapeР
9random_flip_22/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9random_flip_22/random_flip_left_right/strided_slice/stackФ
;random_flip_22/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip_22/random_flip_left_right/strided_slice/stack_1Ф
;random_flip_22/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip_22/random_flip_left_right/strided_slice/stack_2Ц
3random_flip_22/random_flip_left_right/strided_sliceStridedSlice4random_flip_22/random_flip_left_right/Shape:output:0Brandom_flip_22/random_flip_left_right/strided_slice/stack:output:0Drandom_flip_22/random_flip_left_right/strided_slice/stack_1:output:0Drandom_flip_22/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3random_flip_22/random_flip_left_right/strided_sliceь
:random_flip_22/random_flip_left_right/random_uniform/shapePack<random_flip_22/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2<
:random_flip_22/random_flip_left_right/random_uniform/shapeЙ
8random_flip_22/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2:
8random_flip_22/random_flip_left_right/random_uniform/minЙ
8random_flip_22/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8random_flip_22/random_flip_left_right/random_uniform/max
Brandom_flip_22/random_flip_left_right/random_uniform/RandomUniformRandomUniformCrandom_flip_22/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02D
Brandom_flip_22/random_flip_left_right/random_uniform/RandomUniformЙ
8random_flip_22/random_flip_left_right/random_uniform/MulMulKrandom_flip_22/random_flip_left_right/random_uniform/RandomUniform:output:0Arandom_flip_22/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2:
8random_flip_22/random_flip_left_right/random_uniform/MulА
5random_flip_22/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5random_flip_22/random_flip_left_right/Reshape/shape/1А
5random_flip_22/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :27
5random_flip_22/random_flip_left_right/Reshape/shape/2А
5random_flip_22/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :27
5random_flip_22/random_flip_left_right/Reshape/shape/3
3random_flip_22/random_flip_left_right/Reshape/shapePack<random_flip_22/random_flip_left_right/strided_slice:output:0>random_flip_22/random_flip_left_right/Reshape/shape/1:output:0>random_flip_22/random_flip_left_right/Reshape/shape/2:output:0>random_flip_22/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:25
3random_flip_22/random_flip_left_right/Reshape/shape
-random_flip_22/random_flip_left_right/ReshapeReshape<random_flip_22/random_flip_left_right/random_uniform/Mul:z:0<random_flip_22/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2/
-random_flip_22/random_flip_left_right/Reshapeе
+random_flip_22/random_flip_left_right/RoundRound6random_flip_22/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2-
+random_flip_22/random_flip_left_right/RoundЖ
4random_flip_22/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:26
4random_flip_22/random_flip_left_right/ReverseV2/axis­
/random_flip_22/random_flip_left_right/ReverseV2	ReverseV2Arandom_flip_22/random_flip_left_right/control_dependency:output:0=random_flip_22/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ21
/random_flip_22/random_flip_left_right/ReverseV2
)random_flip_22/random_flip_left_right/mulMul/random_flip_22/random_flip_left_right/Round:y:08random_flip_22/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2+
)random_flip_22/random_flip_left_right/mul
+random_flip_22/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+random_flip_22/random_flip_left_right/sub/xў
)random_flip_22/random_flip_left_right/subSub4random_flip_22/random_flip_left_right/sub/x:output:0/random_flip_22/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2+
)random_flip_22/random_flip_left_right/sub
+random_flip_22/random_flip_left_right/mul_1Mul-random_flip_22/random_flip_left_right/sub:z:0Arandom_flip_22/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2-
+random_flip_22/random_flip_left_right/mul_1ћ
)random_flip_22/random_flip_left_right/addAddV2-random_flip_22/random_flip_left_right/mul:z:0/random_flip_22/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2+
)random_flip_22/random_flip_left_right/addЃ
5random_flip_22/random_flip_up_down/control_dependencyIdentity-random_flip_22/random_flip_left_right/add:z:0*
T0*<
_class2
0.loc:@random_flip_22/random_flip_left_right/add*1
_output_shapes
:џџџџџџџџџ27
5random_flip_22/random_flip_up_down/control_dependencyТ
(random_flip_22/random_flip_up_down/ShapeShape>random_flip_22/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:2*
(random_flip_22/random_flip_up_down/ShapeК
6random_flip_22/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_flip_22/random_flip_up_down/strided_slice/stackО
8random_flip_22/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip_22/random_flip_up_down/strided_slice/stack_1О
8random_flip_22/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip_22/random_flip_up_down/strided_slice/stack_2Д
0random_flip_22/random_flip_up_down/strided_sliceStridedSlice1random_flip_22/random_flip_up_down/Shape:output:0?random_flip_22/random_flip_up_down/strided_slice/stack:output:0Arandom_flip_22/random_flip_up_down/strided_slice/stack_1:output:0Arandom_flip_22/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_flip_22/random_flip_up_down/strided_sliceу
7random_flip_22/random_flip_up_down/random_uniform/shapePack9random_flip_22/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:29
7random_flip_22/random_flip_up_down/random_uniform/shapeГ
5random_flip_22/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5random_flip_22/random_flip_up_down/random_uniform/minГ
5random_flip_22/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5random_flip_22/random_flip_up_down/random_uniform/max
?random_flip_22/random_flip_up_down/random_uniform/RandomUniformRandomUniform@random_flip_22/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02A
?random_flip_22/random_flip_up_down/random_uniform/RandomUniform­
5random_flip_22/random_flip_up_down/random_uniform/MulMulHrandom_flip_22/random_flip_up_down/random_uniform/RandomUniform:output:0>random_flip_22/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:џџџџџџџџџ27
5random_flip_22/random_flip_up_down/random_uniform/MulЊ
2random_flip_22/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip_22/random_flip_up_down/Reshape/shape/1Њ
2random_flip_22/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip_22/random_flip_up_down/Reshape/shape/2Њ
2random_flip_22/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip_22/random_flip_up_down/Reshape/shape/3
0random_flip_22/random_flip_up_down/Reshape/shapePack9random_flip_22/random_flip_up_down/strided_slice:output:0;random_flip_22/random_flip_up_down/Reshape/shape/1:output:0;random_flip_22/random_flip_up_down/Reshape/shape/2:output:0;random_flip_22/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:22
0random_flip_22/random_flip_up_down/Reshape/shape
*random_flip_22/random_flip_up_down/ReshapeReshape9random_flip_22/random_flip_up_down/random_uniform/Mul:z:09random_flip_22/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2,
*random_flip_22/random_flip_up_down/ReshapeЬ
(random_flip_22/random_flip_up_down/RoundRound3random_flip_22/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2*
(random_flip_22/random_flip_up_down/RoundА
1random_flip_22/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:23
1random_flip_22/random_flip_up_down/ReverseV2/axisЁ
,random_flip_22/random_flip_up_down/ReverseV2	ReverseV2>random_flip_22/random_flip_up_down/control_dependency:output:0:random_flip_22/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2.
,random_flip_22/random_flip_up_down/ReverseV2ј
&random_flip_22/random_flip_up_down/mulMul,random_flip_22/random_flip_up_down/Round:y:05random_flip_22/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2(
&random_flip_22/random_flip_up_down/mul
(random_flip_22/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(random_flip_22/random_flip_up_down/sub/xђ
&random_flip_22/random_flip_up_down/subSub1random_flip_22/random_flip_up_down/sub/x:output:0,random_flip_22/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2(
&random_flip_22/random_flip_up_down/sub
(random_flip_22/random_flip_up_down/mul_1Mul*random_flip_22/random_flip_up_down/sub:z:0>random_flip_22/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:џџџџџџџџџ2*
(random_flip_22/random_flip_up_down/mul_1я
&random_flip_22/random_flip_up_down/addAddV2*random_flip_22/random_flip_up_down/mul:z:0,random_flip_22/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:џџџџџџџџџ2(
&random_flip_22/random_flip_up_down/add
random_rotation_22/ShapeShape*random_flip_22/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2
random_rotation_22/Shape
&random_rotation_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_rotation_22/strided_slice/stack
(random_rotation_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice/stack_1
(random_rotation_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice/stack_2д
 random_rotation_22/strided_sliceStridedSlice!random_rotation_22/Shape:output:0/random_rotation_22/strided_slice/stack:output:01random_rotation_22/strided_slice/stack_1:output:01random_rotation_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 random_rotation_22/strided_slice
(random_rotation_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice_1/stackЂ
*random_rotation_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_1/stack_1Ђ
*random_rotation_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_1/stack_2о
"random_rotation_22/strided_slice_1StridedSlice!random_rotation_22/Shape:output:01random_rotation_22/strided_slice_1/stack:output:03random_rotation_22/strided_slice_1/stack_1:output:03random_rotation_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_rotation_22/strided_slice_1
random_rotation_22/CastCast+random_rotation_22/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_22/Cast
(random_rotation_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_22/strided_slice_2/stackЂ
*random_rotation_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_2/stack_1Ђ
*random_rotation_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_22/strided_slice_2/stack_2о
"random_rotation_22/strided_slice_2StridedSlice!random_rotation_22/Shape:output:01random_rotation_22/strided_slice_2/stack:output:03random_rotation_22/strided_slice_2/stack_1:output:03random_rotation_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_rotation_22/strided_slice_2
random_rotation_22/Cast_1Cast+random_rotation_22/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_22/Cast_1З
)random_rotation_22/stateful_uniform/shapePack)random_rotation_22/strided_slice:output:0*
N*
T0*
_output_shapes
:2+
)random_rotation_22/stateful_uniform/shape
'random_rotation_22/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|й Р2)
'random_rotation_22/stateful_uniform/min
'random_rotation_22/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|й @2)
'random_rotation_22/stateful_uniform/maxР
=random_rotation_22/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2?
=random_rotation_22/stateful_uniform/StatefulUniform/algorithmя
3random_rotation_22/stateful_uniform/StatefulUniformStatefulUniform<random_rotation_22_stateful_uniform_statefuluniform_resourceFrandom_rotation_22/stateful_uniform/StatefulUniform/algorithm:output:02random_rotation_22/stateful_uniform/shape:output:0*#
_output_shapes
:џџџџџџџџџ*
shape_dtype025
3random_rotation_22/stateful_uniform/StatefulUniformо
'random_rotation_22/stateful_uniform/subSub0random_rotation_22/stateful_uniform/max:output:00random_rotation_22/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2)
'random_rotation_22/stateful_uniform/subђ
'random_rotation_22/stateful_uniform/mulMul<random_rotation_22/stateful_uniform/StatefulUniform:output:0+random_rotation_22/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2)
'random_rotation_22/stateful_uniform/mulо
#random_rotation_22/stateful_uniformAdd+random_rotation_22/stateful_uniform/mul:z:00random_rotation_22/stateful_uniform/min:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2%
#random_rotation_22/stateful_uniform
(random_rotation_22/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(random_rotation_22/rotation_matrix/sub/yЪ
&random_rotation_22/rotation_matrix/subSubrandom_rotation_22/Cast_1:y:01random_rotation_22/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2(
&random_rotation_22/rotation_matrix/subЎ
&random_rotation_22/rotation_matrix/CosCos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/Cos
*random_rotation_22/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_1/yа
(random_rotation_22/rotation_matrix/sub_1Subrandom_rotation_22/Cast_1:y:03random_rotation_22/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_1п
&random_rotation_22/rotation_matrix/mulMul*random_rotation_22/rotation_matrix/Cos:y:0,random_rotation_22/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/mulЎ
&random_rotation_22/rotation_matrix/SinSin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/Sin
*random_rotation_22/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_2/yЮ
(random_rotation_22/rotation_matrix/sub_2Subrandom_rotation_22/Cast:y:03random_rotation_22/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_2у
(random_rotation_22/rotation_matrix/mul_1Mul*random_rotation_22/rotation_matrix/Sin:y:0,random_rotation_22/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/mul_1у
(random_rotation_22/rotation_matrix/sub_3Sub*random_rotation_22/rotation_matrix/mul:z:0,random_rotation_22/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/sub_3у
(random_rotation_22/rotation_matrix/sub_4Sub*random_rotation_22/rotation_matrix/sub:z:0,random_rotation_22/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/sub_4Ё
,random_rotation_22/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,random_rotation_22/rotation_matrix/truediv/yі
*random_rotation_22/rotation_matrix/truedivRealDiv,random_rotation_22/rotation_matrix/sub_4:z:05random_rotation_22/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2,
*random_rotation_22/rotation_matrix/truediv
*random_rotation_22/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_5/yЮ
(random_rotation_22/rotation_matrix/sub_5Subrandom_rotation_22/Cast:y:03random_rotation_22/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_5В
(random_rotation_22/rotation_matrix/Sin_1Sin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Sin_1
*random_rotation_22/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_6/yа
(random_rotation_22/rotation_matrix/sub_6Subrandom_rotation_22/Cast_1:y:03random_rotation_22/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_6х
(random_rotation_22/rotation_matrix/mul_2Mul,random_rotation_22/rotation_matrix/Sin_1:y:0,random_rotation_22/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/mul_2В
(random_rotation_22/rotation_matrix/Cos_1Cos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Cos_1
*random_rotation_22/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_22/rotation_matrix/sub_7/yЮ
(random_rotation_22/rotation_matrix/sub_7Subrandom_rotation_22/Cast:y:03random_rotation_22/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_22/rotation_matrix/sub_7х
(random_rotation_22/rotation_matrix/mul_3Mul,random_rotation_22/rotation_matrix/Cos_1:y:0,random_rotation_22/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/mul_3у
&random_rotation_22/rotation_matrix/addAddV2,random_rotation_22/rotation_matrix/mul_2:z:0,random_rotation_22/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/addу
(random_rotation_22/rotation_matrix/sub_8Sub,random_rotation_22/rotation_matrix/sub_5:z:0*random_rotation_22/rotation_matrix/add:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/sub_8Ѕ
.random_rotation_22/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.random_rotation_22/rotation_matrix/truediv_1/yќ
,random_rotation_22/rotation_matrix/truediv_1RealDiv,random_rotation_22/rotation_matrix/sub_8:z:07random_rotation_22/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2.
,random_rotation_22/rotation_matrix/truediv_1Ћ
(random_rotation_22/rotation_matrix/ShapeShape'random_rotation_22/stateful_uniform:z:0*
T0*
_output_shapes
:2*
(random_rotation_22/rotation_matrix/ShapeК
6random_rotation_22/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_rotation_22/rotation_matrix/strided_slice/stackО
8random_rotation_22/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_22/rotation_matrix/strided_slice/stack_1О
8random_rotation_22/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_22/rotation_matrix/strided_slice/stack_2Д
0random_rotation_22/rotation_matrix/strided_sliceStridedSlice1random_rotation_22/rotation_matrix/Shape:output:0?random_rotation_22/rotation_matrix/strided_slice/stack:output:0Arandom_rotation_22/rotation_matrix/strided_slice/stack_1:output:0Arandom_rotation_22/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_rotation_22/rotation_matrix/strided_sliceВ
(random_rotation_22/rotation_matrix/Cos_2Cos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Cos_2Х
8random_rotation_22/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_1/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_1/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_1/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_1StridedSlice,random_rotation_22/rotation_matrix/Cos_2:y:0Arandom_rotation_22/rotation_matrix/strided_slice_1/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_1/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_1В
(random_rotation_22/rotation_matrix/Sin_2Sin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Sin_2Х
8random_rotation_22/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_2/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_2/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_2/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_2StridedSlice,random_rotation_22/rotation_matrix/Sin_2:y:0Arandom_rotation_22/rotation_matrix/strided_slice_2/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_2/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_2Ц
&random_rotation_22/rotation_matrix/NegNeg;random_rotation_22/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&random_rotation_22/rotation_matrix/NegХ
8random_rotation_22/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_3/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_3/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_3/stack_2ы
2random_rotation_22/rotation_matrix/strided_slice_3StridedSlice.random_rotation_22/rotation_matrix/truediv:z:0Arandom_rotation_22/rotation_matrix/strided_slice_3/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_3/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_3В
(random_rotation_22/rotation_matrix/Sin_3Sin'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Sin_3Х
8random_rotation_22/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_4/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_4/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_4/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_4StridedSlice,random_rotation_22/rotation_matrix/Sin_3:y:0Arandom_rotation_22/rotation_matrix/strided_slice_4/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_4/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_4В
(random_rotation_22/rotation_matrix/Cos_3Cos'random_rotation_22/stateful_uniform:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/Cos_3Х
8random_rotation_22/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_5/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_5/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_5/stack_2щ
2random_rotation_22/rotation_matrix/strided_slice_5StridedSlice,random_rotation_22/rotation_matrix/Cos_3:y:0Arandom_rotation_22/rotation_matrix/strided_slice_5/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_5/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_5Х
8random_rotation_22/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_22/rotation_matrix/strided_slice_6/stackЩ
:random_rotation_22/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_22/rotation_matrix/strided_slice_6/stack_1Щ
:random_rotation_22/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_22/rotation_matrix/strided_slice_6/stack_2э
2random_rotation_22/rotation_matrix/strided_slice_6StridedSlice0random_rotation_22/rotation_matrix/truediv_1:z:0Arandom_rotation_22/rotation_matrix/strided_slice_6/stack:output:0Crandom_rotation_22/rotation_matrix/strided_slice_6/stack_1:output:0Crandom_rotation_22/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_22/rotation_matrix/strided_slice_6Ђ
.random_rotation_22/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation_22/rotation_matrix/zeros/mul/yј
,random_rotation_22/rotation_matrix/zeros/mulMul9random_rotation_22/rotation_matrix/strided_slice:output:07random_rotation_22/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2.
,random_rotation_22/rotation_matrix/zeros/mulЅ
/random_rotation_22/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш21
/random_rotation_22/rotation_matrix/zeros/Less/yѓ
-random_rotation_22/rotation_matrix/zeros/LessLess0random_rotation_22/rotation_matrix/zeros/mul:z:08random_rotation_22/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2/
-random_rotation_22/rotation_matrix/zeros/LessЈ
1random_rotation_22/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :23
1random_rotation_22/rotation_matrix/zeros/packed/1
/random_rotation_22/rotation_matrix/zeros/packedPack9random_rotation_22/rotation_matrix/strided_slice:output:0:random_rotation_22/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:21
/random_rotation_22/rotation_matrix/zeros/packedЅ
.random_rotation_22/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.random_rotation_22/rotation_matrix/zeros/Const
(random_rotation_22/rotation_matrix/zerosFill8random_rotation_22/rotation_matrix/zeros/packed:output:07random_rotation_22/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(random_rotation_22/rotation_matrix/zerosЂ
.random_rotation_22/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation_22/rotation_matrix/concat/axisц
)random_rotation_22/rotation_matrix/concatConcatV2;random_rotation_22/rotation_matrix/strided_slice_1:output:0*random_rotation_22/rotation_matrix/Neg:y:0;random_rotation_22/rotation_matrix/strided_slice_3:output:0;random_rotation_22/rotation_matrix/strided_slice_4:output:0;random_rotation_22/rotation_matrix/strided_slice_5:output:0;random_rotation_22/rotation_matrix/strided_slice_6:output:01random_rotation_22/rotation_matrix/zeros:output:07random_rotation_22/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2+
)random_rotation_22/rotation_matrix/concatЂ
"random_rotation_22/transform/ShapeShape*random_flip_22/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2$
"random_rotation_22/transform/ShapeЎ
0random_rotation_22/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0random_rotation_22/transform/strided_slice/stackВ
2random_rotation_22/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2random_rotation_22/transform/strided_slice/stack_1В
2random_rotation_22/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2random_rotation_22/transform/strided_slice/stack_2ќ
*random_rotation_22/transform/strided_sliceStridedSlice+random_rotation_22/transform/Shape:output:09random_rotation_22/transform/strided_slice/stack:output:0;random_rotation_22/transform/strided_slice/stack_1:output:0;random_rotation_22/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*random_rotation_22/transform/strided_slice
'random_rotation_22/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_rotation_22/transform/fill_valueЬ
7random_rotation_22/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3*random_flip_22/random_flip_up_down/add:z:02random_rotation_22/rotation_matrix/concat:output:03random_rotation_22/transform/strided_slice:output:00random_rotation_22/transform/fill_value:output:0*1
_output_shapes
:џџџџџџџџџ*
dtype0*
	fill_mode	NEAREST*
interpolation
BILINEAR29
7random_rotation_22/transform/ImageProjectiveTransformV3Г
conv2d_81/Conv2D/ReadVariableOpReadVariableOp(conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_81/Conv2D/ReadVariableOp
conv2d_81/Conv2DConv2DLrandom_rotation_22/transform/ImageProjectiveTransformV3:transformed_images:0'conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў *
paddingVALID*
strides
2
conv2d_81/Conv2DЊ
 conv2d_81/BiasAdd/ReadVariableOpReadVariableOp)conv2d_81_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_81/BiasAdd/ReadVariableOpВ
conv2d_81/BiasAddBiasAddconv2d_81/Conv2D:output:0(conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2
conv2d_81/BiasAdd
conv2d_81/SigmoidSigmoidconv2d_81/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2
conv2d_81/Sigmoid
conv2d_81/mulMulconv2d_81/BiasAdd:output:0conv2d_81/Sigmoid:y:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2
conv2d_81/mul
conv2d_81/IdentityIdentityconv2d_81/mul:z:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2
conv2d_81/Identityё
conv2d_81/IdentityN	IdentityNconv2d_81/mul:z:0conv2d_81/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807520*N
_output_shapes<
::џџџџџџџџџўў :џџџџџџџџџўў 2
conv2d_81/IdentityNЬ
max_pooling2d_81/MaxPoolMaxPoolconv2d_81/IdentityN:output:0*1
_output_shapes
:џџџџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_81/MaxPoolГ
conv2d_82/Conv2D/ReadVariableOpReadVariableOp(conv2d_82_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_82/Conv2D/ReadVariableOpп
conv2d_82/Conv2DConv2D!max_pooling2d_81/MaxPool:output:0'conv2d_82/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ *
paddingVALID*
strides
2
conv2d_82/Conv2DЊ
 conv2d_82/BiasAdd/ReadVariableOpReadVariableOp)conv2d_82_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_82/BiasAdd/ReadVariableOpВ
conv2d_82/BiasAddBiasAddconv2d_82/Conv2D:output:0(conv2d_82/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2
conv2d_82/BiasAdd
conv2d_82/SigmoidSigmoidconv2d_82/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2
conv2d_82/Sigmoid
conv2d_82/mulMulconv2d_82/BiasAdd:output:0conv2d_82/Sigmoid:y:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2
conv2d_82/mul
conv2d_82/IdentityIdentityconv2d_82/mul:z:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2
conv2d_82/Identityё
conv2d_82/IdentityN	IdentityNconv2d_82/mul:z:0conv2d_82/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807533*N
_output_shapes<
::џџџџџџџџџ§§ :џџџџџџџџџ§§ 2
conv2d_82/IdentityNЪ
max_pooling2d_82/MaxPoolMaxPoolconv2d_82/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ~~ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_82/MaxPoolГ
conv2d_83/Conv2D/ReadVariableOpReadVariableOp(conv2d_83_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_83/Conv2D/ReadVariableOpн
conv2d_83/Conv2DConv2D!max_pooling2d_82/MaxPool:output:0'conv2d_83/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ||@*
paddingVALID*
strides
2
conv2d_83/Conv2DЊ
 conv2d_83/BiasAdd/ReadVariableOpReadVariableOp)conv2d_83_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_83/BiasAdd/ReadVariableOpА
conv2d_83/BiasAddBiasAddconv2d_83/Conv2D:output:0(conv2d_83/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2
conv2d_83/BiasAdd
conv2d_83/SigmoidSigmoidconv2d_83/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2
conv2d_83/Sigmoid
conv2d_83/mulMulconv2d_83/BiasAdd:output:0conv2d_83/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2
conv2d_83/mul
conv2d_83/IdentityIdentityconv2d_83/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2
conv2d_83/Identityэ
conv2d_83/IdentityN	IdentityNconv2d_83/mul:z:0conv2d_83/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807546*J
_output_shapes8
6:џџџџџџџџџ||@:џџџџџџџџџ||@2
conv2d_83/IdentityNЪ
max_pooling2d_83/MaxPoolMaxPoolconv2d_83/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ>>@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_83/MaxPoolГ
conv2d_84/Conv2D/ReadVariableOpReadVariableOp(conv2d_84_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_84/Conv2D/ReadVariableOpн
conv2d_84/Conv2DConv2D!max_pooling2d_83/MaxPool:output:0'conv2d_84/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@*
paddingVALID*
strides
2
conv2d_84/Conv2DЊ
 conv2d_84/BiasAdd/ReadVariableOpReadVariableOp)conv2d_84_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_84/BiasAdd/ReadVariableOpА
conv2d_84/BiasAddBiasAddconv2d_84/Conv2D:output:0(conv2d_84/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
conv2d_84/BiasAdd
conv2d_84/SigmoidSigmoidconv2d_84/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
conv2d_84/Sigmoid
conv2d_84/mulMulconv2d_84/BiasAdd:output:0conv2d_84/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
conv2d_84/mul
conv2d_84/IdentityIdentityconv2d_84/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ<<@2
conv2d_84/Identityэ
conv2d_84/IdentityN	IdentityNconv2d_84/mul:z:0conv2d_84/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807559*J
_output_shapes8
6:џџџџџџџџџ<<@:џџџџџџџџџ<<@2
conv2d_84/IdentityNЪ
max_pooling2d_84/MaxPoolMaxPoolconv2d_84/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_84/MaxPool
spatial_dropout2d_22/ShapeShape!max_pooling2d_84/MaxPool:output:0*
T0*
_output_shapes
:2
spatial_dropout2d_22/Shape
(spatial_dropout2d_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(spatial_dropout2d_22/strided_slice/stackЂ
*spatial_dropout2d_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_22/strided_slice/stack_1Ђ
*spatial_dropout2d_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_22/strided_slice/stack_2р
"spatial_dropout2d_22/strided_sliceStridedSlice#spatial_dropout2d_22/Shape:output:01spatial_dropout2d_22/strided_slice/stack:output:03spatial_dropout2d_22/strided_slice/stack_1:output:03spatial_dropout2d_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"spatial_dropout2d_22/strided_sliceЂ
*spatial_dropout2d_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_22/strided_slice_1/stackІ
,spatial_dropout2d_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,spatial_dropout2d_22/strided_slice_1/stack_1І
,spatial_dropout2d_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,spatial_dropout2d_22/strided_slice_1/stack_2ъ
$spatial_dropout2d_22/strided_slice_1StridedSlice#spatial_dropout2d_22/Shape:output:03spatial_dropout2d_22/strided_slice_1/stack:output:05spatial_dropout2d_22/strided_slice_1/stack_1:output:05spatial_dropout2d_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$spatial_dropout2d_22/strided_slice_1
"spatial_dropout2d_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"spatial_dropout2d_22/dropout/Constе
 spatial_dropout2d_22/dropout/MulMul!max_pooling2d_84/MaxPool:output:0+spatial_dropout2d_22/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2"
 spatial_dropout2d_22/dropout/MulЌ
3spatial_dropout2d_22/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3spatial_dropout2d_22/dropout/random_uniform/shape/1Ќ
3spatial_dropout2d_22/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :25
3spatial_dropout2d_22/dropout/random_uniform/shape/2є
1spatial_dropout2d_22/dropout/random_uniform/shapePack+spatial_dropout2d_22/strided_slice:output:0<spatial_dropout2d_22/dropout/random_uniform/shape/1:output:0<spatial_dropout2d_22/dropout/random_uniform/shape/2:output:0-spatial_dropout2d_22/strided_slice_1:output:0*
N*
T0*
_output_shapes
:23
1spatial_dropout2d_22/dropout/random_uniform/shape
9spatial_dropout2d_22/dropout/random_uniform/RandomUniformRandomUniform:spatial_dropout2d_22/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
dtype02;
9spatial_dropout2d_22/dropout/random_uniform/RandomUniform
+spatial_dropout2d_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+spatial_dropout2d_22/dropout/GreaterEqual/yЃ
)spatial_dropout2d_22/dropout/GreaterEqualGreaterEqualBspatial_dropout2d_22/dropout/random_uniform/RandomUniform:output:04spatial_dropout2d_22/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2+
)spatial_dropout2d_22/dropout/GreaterEqualЯ
!spatial_dropout2d_22/dropout/CastCast-spatial_dropout2d_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2#
!spatial_dropout2d_22/dropout/Castж
"spatial_dropout2d_22/dropout/Mul_1Mul$spatial_dropout2d_22/dropout/Mul:z:0%spatial_dropout2d_22/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@2$
"spatial_dropout2d_22/dropout/Mul_1Й
2global_average_pooling2d_22/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      24
2global_average_pooling2d_22/Mean/reduction_indicesу
 global_average_pooling2d_22/MeanMean&spatial_dropout2d_22/dropout/Mul_1:z:0;global_average_pooling2d_22/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2"
 global_average_pooling2d_22/MeanЉ
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_96/MatMul/ReadVariableOpВ
dense_96/MatMulMatMul)global_average_pooling2d_22/Mean:output:0&dense_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_96/MatMulЈ
dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_96/BiasAdd/ReadVariableOpІ
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_96/BiasAddt
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_96/ReluЊ
dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_97/MatMul/ReadVariableOpЄ
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_97/MatMulЈ
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_97/BiasAdd/ReadVariableOpІ
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_97/BiasAddt
dense_97/ReluReludense_97/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_97/ReluЉ
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_98/MatMul/ReadVariableOpЃ
dense_98/MatMulMatMuldense_97/Relu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_98/MatMulЇ
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_98/BiasAdd/ReadVariableOpЅ
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_98/BiasAdds
dense_98/ReluReludense_98/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_98/Reluy
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_20/dropout/ConstЉ
dropout_20/dropout/MulMuldense_98/Relu:activations:0!dropout_20/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_20/dropout/Mul
dropout_20/dropout/ShapeShapedense_98/Relu:activations:0*
T0*
_output_shapes
:2
dropout_20/dropout/Shapeе
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype021
/dropout_20/dropout/random_uniform/RandomUniform
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2#
!dropout_20/dropout/GreaterEqual/yъ
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
dropout_20/dropout/GreaterEqual 
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout_20/dropout/CastІ
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_20/dropout/Mul_1Ј
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_99/MatMul/ReadVariableOpЄ
dense_99/MatMulMatMuldropout_20/dropout/Mul_1:z:0&dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_99/MatMulЇ
dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_99/BiasAdd/ReadVariableOpЅ
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_99/BiasAdd|
dense_99/SoftmaxSoftmaxdense_99/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_99/SoftmaxФ
IdentityIdentitydense_99/Softmax:softmax:0!^conv2d_81/BiasAdd/ReadVariableOp ^conv2d_81/Conv2D/ReadVariableOp!^conv2d_82/BiasAdd/ReadVariableOp ^conv2d_82/Conv2D/ReadVariableOp!^conv2d_83/BiasAdd/ReadVariableOp ^conv2d_83/Conv2D/ReadVariableOp!^conv2d_84/BiasAdd/ReadVariableOp ^conv2d_84/Conv2D/ReadVariableOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp ^dense_98/BiasAdd/ReadVariableOp^dense_98/MatMul/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp4^random_rotation_22/stateful_uniform/StatefulUniform*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*t
_input_shapesc
a:џџџџџџџџџ:::::::::::::::::2D
 conv2d_81/BiasAdd/ReadVariableOp conv2d_81/BiasAdd/ReadVariableOp2B
conv2d_81/Conv2D/ReadVariableOpconv2d_81/Conv2D/ReadVariableOp2D
 conv2d_82/BiasAdd/ReadVariableOp conv2d_82/BiasAdd/ReadVariableOp2B
conv2d_82/Conv2D/ReadVariableOpconv2d_82/Conv2D/ReadVariableOp2D
 conv2d_83/BiasAdd/ReadVariableOp conv2d_83/BiasAdd/ReadVariableOp2B
conv2d_83/Conv2D/ReadVariableOpconv2d_83/Conv2D/ReadVariableOp2D
 conv2d_84/BiasAdd/ReadVariableOp conv2d_84/BiasAdd/ReadVariableOp2B
conv2d_84/Conv2D/ReadVariableOpconv2d_84/Conv2D/ReadVariableOp2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2B
dense_98/BiasAdd/ReadVariableOpdense_98/BiasAdd/ReadVariableOp2@
dense_98/MatMul/ReadVariableOpdense_98/MatMul/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp2j
3random_rotation_22/stateful_uniform/StatefulUniform3random_rotation_22/stateful_uniform/StatefulUniform:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н
s
W__inference_global_average_pooling2d_22_layer_call_and_return_conditional_losses_806460

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

р
E__inference_conv2d_82_layer_call_and_return_conditional_losses_807829

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpІ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2	
Sigmoidl
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2

IdentityЩ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-807822*N
_output_shapes<
::џџџџџџџџџ§§ :џџџџџџџџџ§§ 2
	IdentityNЅ

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџ§§ 2

Identity_1"!

identity_1Identity_1:output:0*8
_input_shapes'
%:џџџџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџџџ 
 
_user_specified_nameinputs
Щ
d
F__inference_dropout_20_layer_call_and_return_conditional_losses_808041

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
і	
н
D__inference_dense_99_layer_call_and_return_conditional_losses_806915

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
с
~
)__inference_dense_98_layer_call_fn_808024

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_98_layer_call_and_return_conditional_losses_8068582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В
M
1__inference_max_pooling2d_84_layer_call_fn_806385

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_8063792
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј
o
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807949

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2і
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeд
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yЯ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
В
M
1__inference_max_pooling2d_82_layer_call_fn_806361

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_8063552
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


*__inference_conv2d_83_layer_call_fn_807863

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ||@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_83_layer_call_and_return_conditional_losses_8067032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ||@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ~~ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ~~ 
 
_user_specified_nameinputs

р
E__inference_conv2d_81_layer_call_and_return_conditional_losses_806637

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpІ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2	
Sigmoidl
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2
mule
IdentityIdentitymul:z:0*
T0*1
_output_shapes
:џџџџџџџџџўў 2

IdentityЩ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-806630*N
_output_shapes<
::џџџџџџџџџўў :џџџџџџџџџўў 2
	IdentityNЅ

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџўў 2

Identity_1"!

identity_1Identity_1:output:0*8
_input_shapes'
%:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ќ
р
E__inference_conv2d_83_layer_call_and_return_conditional_losses_806703

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ||@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2	
Sigmoidj
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2
mulc
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ||@2

IdentityХ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-806696*J
_output_shapes8
6:џџџџџџџџџ||@:џџџџџџџџџ||@2
	IdentityNЃ

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ||@2

Identity_1"!

identity_1Identity_1:output:0*6
_input_shapes%
#:џџџџџџџџџ~~ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ~~ 
 
_user_specified_nameinputs
є	
н
D__inference_dense_96_layer_call_and_return_conditional_losses_806804

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_806367

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Я
serving_defaultЛ
_
random_flip_22_inputG
&serving_default_random_flip_22_input:0џџџџџџџџџ<
dense_990
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Ьх
№v
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
ъ_default_save_signature
ы__call__
+ь&call_and_return_all_conditional_losses"юq
_tf_keras_sequentialЯq{"class_name": "Sequential", "name": "sequential_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_22", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_22_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 1]}, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation_22", "trainable": true, "dtype": "float32", "factor": 0.4, "fill_mode": "nearest", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_81", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_81", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_82", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_82", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_83", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_83", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_84", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_84", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d_22", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_96", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_97", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_98", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_99", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 512, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_22", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_22_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 1]}, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation_22", "trainable": true, "dtype": "float32", "factor": 0.4, "fill_mode": "nearest", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_81", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_81", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_82", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_82", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_83", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_83", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_84", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_84", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d_22", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_96", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_97", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_98", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_99", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}

_rng
	keras_api"ј
_tf_keras_layerо{"class_name": "RandomFlip", "name": "random_flip_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 1]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "random_flip_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 1]}, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
П
_rng
	keras_api"Ѓ
_tf_keras_layer{"class_name": "RandomRotation", "name": "random_rotation_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "random_rotation_22", "trainable": true, "dtype": "float32", "factor": 0.4, "fill_mode": "nearest", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}
ј	

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"б
_tf_keras_layerЗ{"class_name": "Conv2D", "name": "conv2d_81", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_81", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 512, 1]}}

"regularization_losses
#trainable_variables
$	variables
%	keras_api
я__call__
+№&call_and_return_all_conditional_losses"ђ
_tf_keras_layerи{"class_name": "MaxPooling2D", "name": "max_pooling2d_81", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_81", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
њ	

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
ё__call__
+ђ&call_and_return_all_conditional_losses"г
_tf_keras_layerЙ{"class_name": "Conv2D", "name": "conv2d_82", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_82", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 255, 255, 32]}}

,regularization_losses
-trainable_variables
.	variables
/	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses"ђ
_tf_keras_layerи{"class_name": "MaxPooling2D", "name": "max_pooling2d_82", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_82", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
њ	

0kernel
1bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
ѕ__call__
+і&call_and_return_all_conditional_losses"г
_tf_keras_layerЙ{"class_name": "Conv2D", "name": "conv2d_83", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_83", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 126, 126, 32]}}

6regularization_losses
7trainable_variables
8	variables
9	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses"ђ
_tf_keras_layerи{"class_name": "MaxPooling2D", "name": "max_pooling2d_83", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_83", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ј	

:kernel
;bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses"б
_tf_keras_layerЗ{"class_name": "Conv2D", "name": "conv2d_84", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_84", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62, 62, 64]}}

@regularization_losses
Atrainable_variables
B	variables
C	keras_api
ћ__call__
+ќ&call_and_return_all_conditional_losses"ђ
_tf_keras_layerи{"class_name": "MaxPooling2D", "name": "max_pooling2d_84", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_84", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
§__call__
+ў&call_and_return_all_conditional_losses"
_tf_keras_layerы{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout2d_22", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
џ__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer№{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ѕ

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
__call__
+&call_and_return_all_conditional_losses"Ю
_tf_keras_layerД{"class_name": "Dense", "name": "dense_96", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_96", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ї

Rkernel
Sbias
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
__call__
+&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense_97", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_97", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
і

Xkernel
Ybias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
__call__
+&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_98", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_98", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
щ
^regularization_losses
_trainable_variables
`	variables
a	keras_api
__call__
+&call_and_return_all_conditional_losses"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
і

bkernel
cbias
dregularization_losses
etrainable_variables
f	variables
g	keras_api
__call__
+&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_99", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_99", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}

hiter

ibeta_1

jbeta_2
	kdecay
llearning_ratemЪmЫ&mЬ'mЭ0mЮ1mЯ:mа;mбLmвMmгRmдSmеXmжYmзbmиcmйvкvл&vм'vн0vо1vп:vр;vсLvтMvуRvфSvхXvцYvчbvшcvщ"
	optimizer
 "
trackable_list_wrapper

0
1
&2
'3
04
15
:6
;7
L8
M9
R10
S11
X12
Y13
b14
c15"
trackable_list_wrapper

0
1
&2
'3
04
15
:6
;7
L8
M9
R10
S11
X12
Y13
b14
c15"
trackable_list_wrapper
Ю
regularization_losses
mlayer_metrics
trainable_variables

nlayers
ometrics
pnon_trainable_variables
	variables
qlayer_regularization_losses
ы__call__
ъ_default_save_signature
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
.
r
_state_var"
_generic_user_object
"
_generic_user_object
.
s
_state_var"
_generic_user_object
"
_generic_user_object
*:( 2conv2d_81/kernel
: 2conv2d_81/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
regularization_losses
tlayer_metrics
trainable_variables

ulayers
vmetrics
wnon_trainable_variables
 	variables
xlayer_regularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
"regularization_losses
ylayer_metrics
#trainable_variables

zlayers
{metrics
|non_trainable_variables
$	variables
}layer_regularization_losses
я__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_82/kernel
: 2conv2d_82/bias
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
Г
(regularization_losses
~layer_metrics
)trainable_variables

layers
metrics
non_trainable_variables
*	variables
 layer_regularization_losses
ё__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
,regularization_losses
layer_metrics
-trainable_variables
layers
metrics
non_trainable_variables
.	variables
 layer_regularization_losses
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_83/kernel
:@2conv2d_83/bias
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
Е
2regularization_losses
layer_metrics
3trainable_variables
layers
metrics
non_trainable_variables
4	variables
 layer_regularization_losses
ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
6regularization_losses
layer_metrics
7trainable_variables
layers
metrics
non_trainable_variables
8	variables
 layer_regularization_losses
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_84/kernel
:@2conv2d_84/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
Е
<regularization_losses
layer_metrics
=trainable_variables
layers
metrics
non_trainable_variables
>	variables
 layer_regularization_losses
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
@regularization_losses
layer_metrics
Atrainable_variables
layers
metrics
non_trainable_variables
B	variables
 layer_regularization_losses
ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Dregularization_losses
layer_metrics
Etrainable_variables
layers
metrics
non_trainable_variables
F	variables
  layer_regularization_losses
§__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Hregularization_losses
Ёlayer_metrics
Itrainable_variables
Ђlayers
Ѓmetrics
Єnon_trainable_variables
J	variables
 Ѕlayer_regularization_losses
џ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 	@2dense_96/kernel
:2dense_96/bias
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
Е
Nregularization_losses
Іlayer_metrics
Otrainable_variables
Їlayers
Јmetrics
Љnon_trainable_variables
P	variables
 Њlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_97/kernel
:2dense_97/bias
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
Е
Tregularization_losses
Ћlayer_metrics
Utrainable_variables
Ќlayers
­metrics
Ўnon_trainable_variables
V	variables
 Џlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 	@2dense_98/kernel
:@2dense_98/bias
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
Е
Zregularization_losses
Аlayer_metrics
[trainable_variables
Бlayers
Вmetrics
Гnon_trainable_variables
\	variables
 Дlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
^regularization_losses
Еlayer_metrics
_trainable_variables
Жlayers
Зmetrics
Иnon_trainable_variables
`	variables
 Йlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_99/kernel
:2dense_99/bias
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
Е
dregularization_losses
Кlayer_metrics
etrainable_variables
Лlayers
Мmetrics
Нnon_trainable_variables
f	variables
 Оlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
0
П0
Р1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	2Variable
:	2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
П

Сtotal

Тcount
У	variables
Ф	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


Хtotal

Цcount
Ч
_fn_kwargs
Ш	variables
Щ	keras_api"П
_tf_keras_metricЄ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
0
С0
Т1"
trackable_list_wrapper
.
У	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Х0
Ц1"
trackable_list_wrapper
.
Ш	variables"
_generic_user_object
/:- 2Adam/conv2d_81/kernel/m
!: 2Adam/conv2d_81/bias/m
/:-  2Adam/conv2d_82/kernel/m
!: 2Adam/conv2d_82/bias/m
/:- @2Adam/conv2d_83/kernel/m
!:@2Adam/conv2d_83/bias/m
/:-@@2Adam/conv2d_84/kernel/m
!:@2Adam/conv2d_84/bias/m
':%	@2Adam/dense_96/kernel/m
!:2Adam/dense_96/bias/m
(:&
2Adam/dense_97/kernel/m
!:2Adam/dense_97/bias/m
':%	@2Adam/dense_98/kernel/m
 :@2Adam/dense_98/bias/m
&:$@2Adam/dense_99/kernel/m
 :2Adam/dense_99/bias/m
/:- 2Adam/conv2d_81/kernel/v
!: 2Adam/conv2d_81/bias/v
/:-  2Adam/conv2d_82/kernel/v
!: 2Adam/conv2d_82/bias/v
/:- @2Adam/conv2d_83/kernel/v
!:@2Adam/conv2d_83/bias/v
/:-@@2Adam/conv2d_84/kernel/v
!:@2Adam/conv2d_84/bias/v
':%	@2Adam/dense_96/kernel/v
!:2Adam/dense_96/bias/v
(:&
2Adam/dense_97/kernel/v
!:2Adam/dense_97/bias/v
':%	@2Adam/dense_98/kernel/v
 :@2Adam/dense_98/bias/v
&:$@2Adam/dense_99/kernel/v
 :2Adam/dense_99/bias/v
і2ѓ
!__inference__wrapped_model_806337Э
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *=Ђ:
85
random_flip_22_inputџџџџџџџџџ
2
.__inference_sequential_22_layer_call_fn_807788
.__inference_sequential_22_layer_call_fn_807313
.__inference_sequential_22_layer_call_fn_807751
.__inference_sequential_22_layer_call_fn_807225Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
I__inference_sequential_22_layer_call_and_return_conditional_losses_807624
I__inference_sequential_22_layer_call_and_return_conditional_losses_806932
I__inference_sequential_22_layer_call_and_return_conditional_losses_807712
I__inference_sequential_22_layer_call_and_return_conditional_losses_806983Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
*__inference_conv2d_81_layer_call_fn_807813Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_81_layer_call_and_return_conditional_losses_807804Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
1__inference_max_pooling2d_81_layer_call_fn_806349р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Д2Б
L__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_806343р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv2d_82_layer_call_fn_807838Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_82_layer_call_and_return_conditional_losses_807829Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
1__inference_max_pooling2d_82_layer_call_fn_806361р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Д2Б
L__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_806355р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv2d_83_layer_call_fn_807863Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_83_layer_call_and_return_conditional_losses_807854Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
1__inference_max_pooling2d_83_layer_call_fn_806373р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Д2Б
L__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_806367р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv2d_84_layer_call_fn_807888Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_84_layer_call_and_return_conditional_losses_807879Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
1__inference_max_pooling2d_84_layer_call_fn_806385р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Д2Б
L__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_806379р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
5__inference_spatial_dropout2d_22_layer_call_fn_807959
5__inference_spatial_dropout2d_22_layer_call_fn_807964
5__inference_spatial_dropout2d_22_layer_call_fn_807921
5__inference_spatial_dropout2d_22_layer_call_fn_807926Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2џ
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807916
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807949
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807954
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807911Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Є2Ё
<__inference_global_average_pooling2d_22_layer_call_fn_806466р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
П2М
W__inference_global_average_pooling2d_22_layer_call_and_return_conditional_losses_806460р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
г2а
)__inference_dense_96_layer_call_fn_807984Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_96_layer_call_and_return_conditional_losses_807975Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_97_layer_call_fn_808004Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_97_layer_call_and_return_conditional_losses_807995Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_98_layer_call_fn_808024Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_98_layer_call_and_return_conditional_losses_808015Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
+__inference_dropout_20_layer_call_fn_808046
+__inference_dropout_20_layer_call_fn_808051Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ъ2Ч
F__inference_dropout_20_layer_call_and_return_conditional_losses_808041
F__inference_dropout_20_layer_call_and_return_conditional_losses_808036Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
г2а
)__inference_dense_99_layer_call_fn_808071Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense_99_layer_call_and_return_conditional_losses_808062Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
иBе
$__inference_signature_wrapper_807360random_flip_22_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 Ж
!__inference__wrapped_model_806337&'01:;LMRSXYbcGЂD
=Ђ:
85
random_flip_22_inputџџџџџџџџџ
Њ "3Њ0
.
dense_99"
dense_99џџџџџџџџџЙ
E__inference_conv2d_81_layer_call_and_return_conditional_losses_807804p9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ "/Ђ,
%"
0џџџџџџџџџўў 
 
*__inference_conv2d_81_layer_call_fn_807813c9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ
Њ ""џџџџџџџџџўў Й
E__inference_conv2d_82_layer_call_and_return_conditional_losses_807829p&'9Ђ6
/Ђ,
*'
inputsџџџџџџџџџџџ 
Њ "/Ђ,
%"
0џџџџџџџџџ§§ 
 
*__inference_conv2d_82_layer_call_fn_807838c&'9Ђ6
/Ђ,
*'
inputsџџџџџџџџџџџ 
Њ ""џџџџџџџџџ§§ Е
E__inference_conv2d_83_layer_call_and_return_conditional_losses_807854l017Ђ4
-Ђ*
(%
inputsџџџџџџџџџ~~ 
Њ "-Ђ*
# 
0џџџџџџџџџ||@
 
*__inference_conv2d_83_layer_call_fn_807863_017Ђ4
-Ђ*
(%
inputsџџџџџџџџџ~~ 
Њ " џџџџџџџџџ||@Е
E__inference_conv2d_84_layer_call_and_return_conditional_losses_807879l:;7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ>>@
Њ "-Ђ*
# 
0џџџџџџџџџ<<@
 
*__inference_conv2d_84_layer_call_fn_807888_:;7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ>>@
Њ " џџџџџџџџџ<<@Ѕ
D__inference_dense_96_layer_call_and_return_conditional_losses_807975]LM/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџ
 }
)__inference_dense_96_layer_call_fn_807984PLM/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџІ
D__inference_dense_97_layer_call_and_return_conditional_losses_807995^RS0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 ~
)__inference_dense_97_layer_call_fn_808004QRS0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЅ
D__inference_dense_98_layer_call_and_return_conditional_losses_808015]XY0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 }
)__inference_dense_98_layer_call_fn_808024PXY0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Є
D__inference_dense_99_layer_call_and_return_conditional_losses_808062\bc/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_dense_99_layer_call_fn_808071Obc/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџІ
F__inference_dropout_20_layer_call_and_return_conditional_losses_808036\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "%Ђ"

0џџџџџџџџџ@
 І
F__inference_dropout_20_layer_call_and_return_conditional_losses_808041\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "%Ђ"

0џџџџџџџџџ@
 ~
+__inference_dropout_20_layer_call_fn_808046O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "џџџџџџџџџ@~
+__inference_dropout_20_layer_call_fn_808051O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "џџџџџџџџџ@р
W__inference_global_average_pooling2d_22_layer_call_and_return_conditional_losses_806460RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 З
<__inference_global_average_pooling2d_22_layer_call_fn_806466wRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџя
L__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_806343RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_max_pooling2d_81_layer_call_fn_806349RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџя
L__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_806355RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_max_pooling2d_82_layer_call_fn_806361RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџя
L__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_806367RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_max_pooling2d_83_layer_call_fn_806373RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџя
L__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_806379RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_max_pooling2d_84_layer_call_fn_806385RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџй
I__inference_sequential_22_layer_call_and_return_conditional_losses_806932s&'01:;LMRSXYbcOЂL
EЂB
85
random_flip_22_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 и
I__inference_sequential_22_layer_call_and_return_conditional_losses_806983&'01:;LMRSXYbcOЂL
EЂB
85
random_flip_22_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ъ
I__inference_sequential_22_layer_call_and_return_conditional_losses_807624}s&'01:;LMRSXYbcAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Щ
I__inference_sequential_22_layer_call_and_return_conditional_losses_807712|&'01:;LMRSXYbcAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 А
.__inference_sequential_22_layer_call_fn_807225~s&'01:;LMRSXYbcOЂL
EЂB
85
random_flip_22_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџЏ
.__inference_sequential_22_layer_call_fn_807313}&'01:;LMRSXYbcOЂL
EЂB
85
random_flip_22_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџЂ
.__inference_sequential_22_layer_call_fn_807751ps&'01:;LMRSXYbcAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЁ
.__inference_sequential_22_layer_call_fn_807788o&'01:;LMRSXYbcAЂ>
7Ђ4
*'
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџб
$__inference_signature_wrapper_807360Ј&'01:;LMRSXYbc_Ђ\
Ђ 
UЊR
P
random_flip_22_input85
random_flip_22_inputџџџџџџџџџ"3Њ0
.
dense_99"
dense_99џџџџџџџџџР
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807911l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@
 Р
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807916l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@
 ї
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807949ЂVЂS
LЂI
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ї
P__inference_spatial_dropout2d_22_layer_call_and_return_conditional_losses_807954ЂVЂS
LЂI
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
5__inference_spatial_dropout2d_22_layer_call_fn_807921_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ " џџџџџџџџџ@
5__inference_spatial_dropout2d_22_layer_call_fn_807926_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ " џџџџџџџџџ@Я
5__inference_spatial_dropout2d_22_layer_call_fn_807959VЂS
LЂI
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЯ
5__inference_spatial_dropout2d_22_layer_call_fn_807964VЂS
LЂI
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ