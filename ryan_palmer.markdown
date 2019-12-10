## Ryan Palmer's tasks
- [ ] 1. DL Workbench
	- [x] 1. Installation via docker
	https://software.intel.com/en-us/openvino-toolkit/documentation/featured
	https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_Workbench.html
	- [ ] 2. trying pre-trained models
	- [ ] 3.  Test the Developer Experience when converting models from their original form (TensorFlow, Caffe, ONYX) to OpenVino format using the Deep Learning Workbench
	- [ ] 4. Convert model from original format to OpenVINO (using the GUI interface of the Deep Learning Workbench tool - (Get training from Mikhail's team).
	- [ ] 5. Quantize the model to INT-8 format
- [ ] 2. Challenge: Understanding model performance. OpenVINO --> should integrate into the workflow of customers
	- [ ] 1. Tools are often cryptic and difficult to understand outside of Intel - or require training. We want to reduce that.  Create tools that any SW developer with basic experience can adopt.Deep Learning models - pretrained.  And process them through OpenVINO - BUT using our DL Workbench program (vs. command line)  
- [x] 3. DevCloud (OpenVINO web version overiew)
	- [x] 1. https://software.intel.com/en-us/iot/home

---

	https://hub.docker.com/r/openvino/workbench

	Help #1.  Test the Developer Experience when converting models from their original form (TensorFlow, Caffe, ONYX) to OpenVino format using the Deep Learning Workbench

	[‎12/‎2/‎2019 9:51 AM]  Palmer, Ryan:  
	1. Convert model from original format to OpenVINO (using the GUI interface of the Deep Learning Workbench tool - (Get training from Mikhail's team).
	2. Quantize the model to INT-8 format

	[‎12/‎2/‎2019 9:52 AM]  Palmer, Ryan:  
	Challenge: Understanding model performance
	OpenVINO --> should integrate into the workflow of customers

	[‎12/‎2/‎2019 9:54 AM]  Palmer, Ryan:  
	Tools are often cryptic and difficult to understand outside of Intel - or require training.

	[‎12/‎2/‎2019 9:54 AM]  Palmer, Ryan:  
	We want to reduce that.  Create tools that any SW developer with basic experience can adopt.

---

- [x] 1.	Create an account with DevCloud to learn how OpenVINO works and experiment with the tutorials for Object Detection and Classification as a starting point.
- [x] 2.	Walk through the Smart Video workshop here on your own laptop/local system: https://github.com/intel-iot-devkit/smart-video-workshop
	- [x] a.	https://github.com/intel-iot-devkit/smart-video-workshop/blob/master/Lab_setup.md
	- [ ] b.	https://github.com/intel-iot-devkit/smart-video-workshop/tree/master/object-detection

- [x] 3.	THEN  once you have a feel for OpenVINO, install the Docker for the Deep Learning Workbench (.   This is an application I will walk through with you.  Im asking for help to try out some of the user flows (wearing the hat of developer) and figure out how to use the application without any instruction.    
- [x] a.	https://hub.docker.com/r/openvino/workbench
- [x] b.	You can reference the documentation here:   https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Introduction.html

If you see any errors or issues with the documentation or encountered stumbling blocks  Please write it down and document a list of issues and questions.   

- [x] 4.	Once you have the Workbench running on your laptop, let me know and well go from there.   Ill walk you through a few scenarios where we need some UX testing done.  




## [DevCloud](https://devcloud.intel.com/edge/get_started/tutorials/)

### Classification

1. What should I do to be sure? How to check those files in Jupyter? Why do I need to do this for online tutorial?  
		Prerequisites
		Before going through this tutorial, please be sure that:

		All files from the .zip file containing the tutorial are present and in the same directory. The required files are:
		- tutorial_classification_sample.ipynb - This Jupyter Notebook
		- squeezenet1.1/squeezenet1.1.bin and squeezenet1.1/squeezenet1.1.xml - The IR files for the   inference model created using Model Optimizer
		- squeezenet1.1/squeezenet1.1.labels.txt - mapping of numerical labels to text strings
		...

2. *Reshape to match input dimensions `frame = frame.reshape((n, c, h, w))`.* Not clear how reshape function work, how old dimensions look like and how new one will. Can you provide some example please, like in other functions Before

3. Didn't get what cpu_extension_path is needed for. Like, are there options? Can we use like different extensions? What are they?
		cpu_extension_path - Path to a shared library with CPU extension kernels for custom layers not already included in plugin

4. Like, are we **creating** a plugin or **loading**? Because I don't understand why do we need to create it, to write from nothing. It's confusing.
		Here we create a plugin object for the specified device using IEPlugin().
		If the plugin is for a CPU device, and the cpu_extensions_path variable has been set, we load the extensions library.

		# create plugin for device
		plugin = IEPlugin(device=device)
		print("A plugin object has been created for device [", plugin.device, "]\n")

		# if the device is CPU and a path to an extension library is set, load the extension library
		if cpu_extension_path and 'CPU' in device:
		    plugin.add_cpu_extension(cpu_extension_path)
		    print("CPU extension [", cpu_extension_path, "] has been loaded")

5. I don't understand why do we need to use those names (more importantly, why are they stored in next(iter(..)))
		# store name of input and output blobs
		input_blob = next(iter(net.inputs))
		output_blob = next(iter(net.outputs))

6. Why use `cap = cv2.VideoCapture(input_path)` to load an image? **Video** Capture.
		Resized input image from (4020, 3124) to (227, 227)
	But the image showed next isn't a square 277x277. It's actually **183x271** pixels

### Object detection

1. Exerciese #3. Here we use loadInput**Image** for the **video** and in the previous guide we used **.Video**Capture method** for the image processing (it's kinda confusing, I thought I can use VideoCapture method as well).

	`print("Loading video [",input_path,"]")`  
	`cap = loadInputImage(input_path)`


## [Smart video Workshop](https://github.com/intel-iot-devkit/smart-video-workshop)

1. Software installation steps require to install **Intel Media SDK** and **OpenCL**. It would be great if there was a quick explanation what for, because people might not want to trash their system with temporarily stuffs which is hard to uninstall. The explanation will give a confidence.

2. Troubles with installing yaml: official [guide's](http://www.linuxfromscratch.org/blfs/view/8.3/general/yaml.html) commands don't work. So, it appears, that [yaml](https://yaml.org) has different packages for different needs: python, c++. And there are like 3 options for the Python. It's a minor gap, but it is. [This guide](https://yaml.readthedocs.io/en/latest/install.html) broke my pip program. [Final solution for yaml](https://stackoverflow.com/a/14262462)

3. Why are there so much requirements I need to install manually? Is it that hard to write a script which checks damned requirements, installs and runs commands?

4. Great [presentation](https://github.com/intel-iot-devkit/smart-video-workshop/blob/master/presentations/01-Introduction-to-Intel-Smart-Video-Tools.pdf), explaining workflow with images, differences, etc. I wish I've seen this before getting familiar with the product

5.  smart-video-workshop-master/dl-model-training/Python/Deep_Learning_Tutorial.ipynb: I didn't get what "Epoches" are, why are there 5 of them.

6. [4. Build the sample application with make file](https://github.com/intel-iot-devkit/smart-video-workshop/tree/master/object-detection#4-build-the-sample-application-with-make-file) gave me  
		main.cpp: In function ‘int main(int, char**)’:
		main.cpp:133:95: warning: ‘InferenceEngine::InferencePlugin InferenceEngine::PluginDispatcher::getPluginByDevice(const string&) const’ is deprecated [-Wdeprecated-declarations]
		 ngine::InferenceEnginePluginPtr _plugin(dispatcher.getPluginByDevice(FLAGS_d));
		                                                                             ^
		In file included from main.cpp:31:0:
		/opt/intel/openvino/deployment_tools/inference_engine/include/ie_plugin_dispatcher.hpp:43:21: note: declared here
		     InferencePlugin getPluginByDevice(const std::string& deviceName) const;
		                     ^~~~~~~~~~~~~~~~~
		main.cpp:170:43: warning: ‘InferenceEngine::SizeVector InferenceEngine::InputInfo::getDims() const’ is deprecated [-Wdeprecated-declarations]
		             inputDims=input_data->getDims();
		                                           ^
		In file included from /opt/intel/openvino/deployment_tools/inference_engine/include/ie_icnn_network.hpp:18:0,
		                 from /opt/intel/openvino/deployment_tools/inference_engine/include/ie_plugin.hpp:11,
		                 from /opt/intel/openvino/deployment_tools/inference_engine/include/details/ie_so_pointer.hpp:14,
		                 from /opt/intel/openvino/deployment_tools/inference_engine/include/ie_plugin_ptr.hpp:11,
		                 from /opt/intel/openvino/deployment_tools/inference_engine/include/ie_plugin_dispatcher.hpp:11,
		                 from main.cpp:31:
		/opt/intel/openvino/deployment_tools/inference_engine/include/ie_input_info.hpp:157:16: note: declared here
		     SizeVector getDims() const {
		                ^~~~~~~_
No progress since then, I dropped it

## [DL Workbench](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_Workbench.html)

### Installation

1. To install via Docker I had to install the Docker itself first. I couldn't run ["hello-world"](https://docs.docker.com/install/linux/docker-ce/ubuntu/) package (no response from a docker server, *probably* proxy issue (an error doesn't tell that it's proxy issue, but my manager gave me the advice)). Had to follow [this guide](https://wiki.ith.intel.com/display/SIPHome/Install+Docker#) to make it run "hello world". Lately I've found how to set proxies [here](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_from_Docker_Hub.html), but I think the information order might be better, because user installs Prerequisites first, then goes for [Install from Docker Hub](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_from_Docker_Hub.html). So, putting the **Prerequisites** directly in the [guide](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_from_Docker_Hub.html) about will improve installation experience. OR remove "Docker*" from **Prerequisites**

2. Then I realized, I've already installed OpenVINO toolkit and the workbench might be there.  
>Run DL Workbench
Open a terminal in the DL Workbench folder. The path to the folder is
/< path_to_installed_package>/deployment_tools/tools/workbench.

	Firstly, I didn't get where installed package placed. It's kinda complicated to work with ubuntu. (The default path is **/opt/intel/openvino/**, I think it must be placed [here](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_from_Package.html) instead of *path_to_installed_package*, because every other pages do so and it's intuitive)

3. **Run DL Workbench**  
Run the following command, specifying the path to the downloaded **archive** (not very clear) with the OpenVINO™ toolkit package:  
	`/opt/intel/openvino/deployment_tools/tools/workbench**$ ./run_openvino_workbench.sh -PACKAGE_PATH ~/Downloads/l_openvino_toolkit_p_2019.3.376.tgz**`  
		Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.40/containers/json?all=1: dial unix /var/run/docker.sock: connect: permission denied  
		/tmp/workbench /opt/intel/openvino/deployment_tools/tools/workbench  
		Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post http://%2Fvar%2Frun%2Fdocker.sock/v1.40/build?buildargs=%7B%22db_password%22%3A%22openvino%22%2C%22http_proxy%22%3A%22http%3A%2F%2Fproxy-chain.intel.com%3A912%22%2C%22https_proxy%22%3A%22http%3A%2F%2Fproxy-chain.intel.com%3A911%22%2C%22no_proxy%22%3A%22intel.com%2C.intel.com%2Clocalhost%2C127.0.0.1%22%2C%22rabbitmq_password%22%3A%22openvino%22%7D&cachefrom=%5B%5D&cgroupparent=&cpuperiod=0&cpuquota=0&cpusetcpus=&cpusetmems=&cpushares=0&dockerfile=Dockerfile&labels=%7B%7D&memory=0&memswap=0&networkmode=default&rm=1&session=orvrmh581616p3p0t4shxtcxz&shmsize=0&t=workbench&target=&ulimits=null&version=1: dial unix /var/run/docker.sock: connect: permission denied

	Did:  
	`trifonov@nnlicv411:~ $ export https_proxy=http://proxy-chain.intel.com:912/`  
	`trifonov@nnlicv411:~ $ export http_proxy=http://proxy-chain.intel.com:911/`  
	`trifonov@nnlicv411:~ $ export no_proxy=localhost,127.0.0.1`

	But still same permission issue. After an hour I've finally found a [FIX](https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue/51362528#51362528). At this point I've already hated my life.  

	New input:  
`trifonov@nnlicv411:/opt/intel/openvino/deployment_tools/tools/workbench$ ./run_openvino_workbench.sh -PACKAGE_PATH ~/Downloads/l_openvino_toolkit_p_2019.3.376.tgz`  

			Notes, that ./run_openvino_workbench says about parameters:
		    -PACKAGE_PATH - path to OpenVINO package for Ubuntu 16 in tar.gz format

			but I have .tgz and it worked fine

	So, why do I need to utilize old archive of openvino? I thought it's needed only to extract files from there (then build and install). I think, this is strange.  
	Output:
		https://salsa.debian.org/opencl-team/ocl-icd.git  
		Please use:  
		git clone https://salsa.debian.org/opencl-team/ocl-icd.git  
		to retrieve the latest (possibly unreleased) updates to the package.  
		Picking 'dpkg' as source package instead of 'dpkg-dev'  
		NOTICE: 'dpkg' packaging is maintained in the 'Git' version control system at:  
		https://git.dpkg.org/git/dpkg/dpkg.git  
		Please use:  
		git clone https://git.dpkg.org/git/dpkg/dpkg.git  
		to retrieE: Can not find version '200+deb10u3' of package 'postgresql'  
		E: Unable to find a source package for postgresql-common  
		ve the latest (possibly unreleased) updates to the package.  
		Picking 'lsb' as source package instead of 'lsb-release'  
		NOTICE: 'lsb' packaging is maintained in the 'Git' version control system at:  
		https://salsa.debian.org/debian/lsb.git  
		Please use:  
		git clone https://salsa.debian.org/debian/lsb.git  
		to retrieve the latest (possibly unreleased) updates to the package.  
		Picking 'ncurses' as source package instead of 'libtinfo5'  
		NOTICE: 'ncurses' packaging is maintained in the 'Git' version control system at:  
		https://salsa.debian.org/debian/ncurses.git  
		Please use:  
		git clone https://salsa.debian.org/debian/ncurses.git  
		to retrieve the latest (possibly unreleased) updates to the package.  
		NOTICE: 'curl' packaging is maintained in the 'Git' version control system at:  
		https://salsa.debian.org/debian/curl.git  
		Please use:  
		git clone https://salsa.debian.org/debian/curl.git  
		to retrieve the latest (possibly unreleased) updates to the package.  
		Picking 'python3-defaults' as source package instead of 'python3'  
		NOTICE: 'python3-defaults' packaging is maintained in the 'Git' version control system at:  
		https://salsa.debian.org/cpython-team/python3-defaults.git  
		Please use:  
		git clone https://salsa.debian.org/cpython-team/python3-defaults.git  
		to retrieve the latest (possibly unreleased) updates to the package.  
		Picking 'python-pip' as source package instead of 'python3-pip'  
		NOTICE: 'python-pip' packaging is maintained in the 'Git' version control system at:  
		https://salsa.debian.org/python-team/modules/python-pip.git  
		Please use:  
		git clone https://salsa.debian.org/python-team/modules/python-pip.git  
		to retrieve the latest (possibly unreleased) updates to the package.  
		Picking 'python-setuptools' as source package instead of 'python3-setuptools'  
		Picking 'python3-defaults' as source package instead of 'python3-dev'  
		NOTICE: 'python3-defaults' packaging is maintained in the 'Git' version control system at:  
		https://salsa.debian.org/cpython-team/python3-defaults.git  
		Please use:  
		git clone https://salsa.debian.org/cpython-team/python3-defaults.git  
		to retrieve the latest (possibly unreleased) updates to the package.  
		Picking 'postgresql-common' as source package instead of 'postgresql'  
		The command '/bin/sh -c sed -i '3ideb-src http://deb.debian.org/debian buster main' /etc/apt/sources.list &&     apt-get update &&     apt-get install -y --no-install-recommends ${OPENVINO_DEPENDENCIES} ${WORKBENCH_DEPENDENCIES} &&     apt-get source ${OPENVINO_DEPENDENCIES} ${WORKBENCH_DEPENDENCIES} &&     rm -rf /var/lib/apt/lists/*' returned a non-zero code: 100  
		* sad* *
	So, there are lots of erros and http://127.0.0.1:5665/ doesn't work. What did I miss? I don't know.  
	Input to run through the docker, placed [here](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_from_Docker_Hub.html):  

	 ``trifonov@nnlicv411:/opt/intel/openvino/deployment_tools/tools/workbench$ docker run -p 127.0.0.1:5665:5665 --name workbench --privileged -v /dev/bus/usb:/dev/bus/usb -v /dev/dri:/dev/dri -it openvino/workbench:latest --build-arg https_proxy=http://proxy-chain.intel.com:912/ --build-arg http_proxy=http://proxy-chain.intel.com:911/ --build-arg no_proxy=127.0.0.0 ``   

	 Output:

		[setupvars.sh] OpenVINO environment initialized  
		[....] Starting PostgreSQL 11 database server: main    
		. ok
		[ ok ] Starting RabbitMQ Messaging Server: rabbitmq-serversdasdasd.
		Adding user "openvino" ...
		Adding vhost "openvino_vhost" ...
		Setting tags for user "openvino" to [openvino_tag] ...
		Setting permissions for user "openvino" in vhost "openvino_vhost" ...
		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint8 = np.dtype([("qint8", np.int8, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint16 = np.dtype([("qint16", np.int16, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint32 = np.dtype([("qint32", np.int32, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  np_resource = np.dtype([("resource", np.ubyte, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint8 = np.dtype([("qint8", np.int8, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint16 = np.dtype([("qint16", np.int16, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint32 = np.dtype([("qint32", np.int32, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  np_resource = np.dtype([("resource", np.ubyte, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint8 = np.dtype([("qint8", np.int8, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint8 = np.dtype([("qint8", np.int8, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint16 = np.dtype([("qint16", np.int16, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint16 = np.dtype([("qint16", np.int16, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint32 = np.dtype([("qint32", np.int32, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint32 = np.dtype([("qint32", np.int32, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  np_resource = np.dtype([("resource", np.ubyte, 1)])

		10:03:23 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  np_resource = np.dtype([("resource", np.ubyte, 1)])

		 -------------- celery@bf05e15e8ebb v4.3.0 (rhubarb)
		---- **** -----
		--- * ***  * -- Linux-5.0.0-37-generic-x86_64-with-debian-10.1 2019-12-05 10:03:25
		-- * - **** ---
		- ** ---------- [config]
		- ** ---------- .> app:         __main__:0x7fe414aa3a20
		- ** ---------- .> transport:   amqp://openvino:**@localhost:5672/openvino_vhost
		- ** ---------- .> results:     rpc://
		- *** --- * --- .> concurrency: 12 (prefork)
		-- ******* ---- .> task events: OFF (enable -E to monitor tasks in this worker)
		--- ***** -----
		 -------------- [queues]
		                .> celery           exchange=celery(direct) key=celery

		[tasks]
		  . app.main.tasks.task.Task
		  . celery.accumulate
		  . celery.backend_cleanup
		  . celery.chain
		  . celery.chord
		  . celery.chord_unlock
		  . celery.chunks
		  . celery.group
		  . celery.map
		  . celery.starmap

		Error: No nodes replied within time constraint.
		Celery is not ready to the moment. Retry in 2 seconds
		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint8 = np.dtype([("qint8", np.int8, 1)])

		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])

		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint16 = np.dtype([("qint16", np.int16, 1)])

		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])

		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint32 = np.dtype([("qint32", np.int32, 1)])

		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  np_resource = np.dtype([("resource", np.ubyte, 1)])

		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint8 = np.dtype([("qint8", np.int8, 1)])

		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])

		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint16 = np.dtype([("qint16", np.int16, 1)])

		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])

		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  _np_qint32 = np.dtype([("qint32", np.int32, 1)])

		10:03:26 accuracy_checker WARNING: /usr/local/lib/python3.7/dist-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
		  np_resource = np.dtype([("resource", np.ubyte, 1)])

		> dl-workbench-proxy-server@0.0.1 start /opt/intel/openvino/deployment_tools/tools/workbench/proxy
		> node proxy-server.js

		Proxy server listening on http://127.0.0.1:5665		__
Why are there so many warnings? Did I fail somewhere? I really don't understand what I did wrong.  

	But ok, let's move on. As being told [here](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_from_Docker_Hub.html) I need to open http://127.0.0.1:5665. It doesn't work. **However** I saw how the hired dev used some wiki on github and he tried this command:  
	`docker run -p 0.0.0.0:5665:5665 --privileged -v /dev/bus/usb:/dev/bus/usb -v /dev/dri:/dev/dri -e PROXY_HOST_ADDRESS=0.0.0.0 -i openvino/workbench`  

		Proxy server listening on http://0.0.0.0:5665

	Ok, I tried to connect http://0.0.0.0:5665, because it supposed to work there, but it didn't. Then I saw connected to 127.0.0.1:5665 and **it worked** for me as well.
	So, in conclusion, I couldn't install it without a help.

4. **Questions**  
So, how do I close workbench? Just CTRL+C? Do I need to do `docker stop workbench`, how to run it again?
https://lookback.io/watch/VQANLLgCQCm5ppX3p

5. **Suggestions**

	1. Move whole [Deep Learning Workbench Developer Installation Guide](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_Workbench.html) to the **GETTING STARTED** section from **GUIDES**, because all installation are there. (Or put all guides in **GUIDES** section, because it's confusing: guides are everywhere)

	2. Give a basic examples with explanations how to work with [Docker](https://docs.docker.com/engine/reference/run/), explain how to use it to maintain the Workbench

	3. After installation, I want to test, to do something really fast and **clear**.
>You have successfully installed the OpenVINO™ DL Workbench. [Move on to Work with Models and Sample Datasets](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Work_with_Models_and_Sample_Datasets.html).

	But this page is kinda empty. It would be better to have examples on the first page with full explanation what now you can do. *Like, give me something interesting already*

	4. To give an explanation what *models*, *samples* are, where can we find it to make quick test and so on. I do understand, that many experienced developers already know that, but you know, if I want to start from nothing, where should I gain such info?

### Working with DL Workbench

1. The [page](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Work_with_Models_and_Sample_Datasets.html) I'm on



## Common questions:

1. What are demos? I tried to run `./human_pose_estimation_demo` and some others, but they require an input video/image (which I do understand) and a model. Moreover, the `./human_pose_estimation_demo` requires `-m Path to the Human Pose Estimation model`, which I don't understand where does it placed and.. Like, what is the purpose of the demo script, if I need to find the model and the input? What the difference between demos? What happened, if I load different model from what expected? How to understand what model is expected?
