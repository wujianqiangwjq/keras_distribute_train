   一直探索keras着分布氏训练,如今遇到一些问题没办法解决，只有到日后解决了
  环境:
	singularity 镜像中,master/worker 每台机器多GPU
 报错:
	INFO:tensorflow:Calling model_fn.
	INFO:tensorflow:Error reported to Coordinator: You must specify an aggregation method to update a MirroredVariable in Tower Context.

	ValueError: You must specify an aggregation method to update a MirroredVariable in Tower Context
