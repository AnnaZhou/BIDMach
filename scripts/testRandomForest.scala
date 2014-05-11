import edu.berkeley.bid.CUMAT
import BIDMach.models.NewRandomForest
import BIDMach.models.RandomForest
import BIDMach.models.RandForest


class testHashNewRandomForest {
   	/**********
	* fdata = nfeats x n 
	* fbounds = nfeats x 2
	* treenodes = ntrees x n
	* cats = ncats x n // labels 1 or greater
	* ntrees = ntrees
	* nsamps = nsamps
	* fieldLengths = 1 x 5 or 5 x 1
	* depth = depth
	* treesMeta = 5 x (ntrees * (2^depth - 1)) // data about each node; row #0: threshold; row #1: infoGain; row #2: cat; row #3: bestFeat; row #4: isLeaf 
	**********/
	val ITree = 0; val INode = 1; val JFeat = 2; val IFeat = 3; val IVFeat = 4; val ICat = 5
	def prepTreeForTrain1 : NewRandomForest = {
		val x : DMat = load("../Data/bidmatSpamData.mat", "Xtrain"); // ntrain x nfeats
		var y : DMat = load("../Data/bidmatSpamData.mat", "ytrain"); // ntrain x 1
		
		val numCats = 2;
		val n = x.nrows

		val fdata : Mat =  FMat(x.t) // nfeats x n
		val cats : Mat = sparse(FMat((icol(0->numCats) * iones(1,n) )  == y.t));  //ncats x n
		val ntrees : Int = 64
		val depth : Int = 13
		val nsamps : Int = 32 //((fdata.nrows * 2f)/3f).toInt
		val useGPU : Boolean = true
		val rF = new NewRandomForest(fdata, cats, ntrees, depth, nsamps, useGPU)
		rF
	}

	/*****
	 * D: 13 T: 60 0.93229
	 *****/
	def testTrain1 : Int = {
		val rF = prepTreeForTrain1
		rF.train
		val xTest : DMat = load("../Data/bidmatSpamData.mat", "Xtest"); // ntest x nfeats
		var yTest : DMat = load("../Data/bidmatSpamData.mat", "ytest"); // ntest x 1
		val yGuess : Mat = rF.classify(FMat(xTest.t)) 
		// println(yGuess)
		println(calcAccuracy(yGuess, FMat(yTest.t)))
		0
	}

	def prepSimpleTrain : NewRandomForest = {
		val x = FMat(1\2\3\4 on 4\7\8\9 on 8\1\12\15 on 5\6\7\8 on 9\1\12\13)
		val y = IMat(1\0\1\0)
		
		val numCats = 2;
		val n = x.ncols

		val fdata : Mat =  FMat(x) // nfeats x n
		val cats : Mat = sparse(FMat((icol(0->numCats) * iones(1,n) )  == y));  //ncats x n
		val ntrees : Int = 1
		val depth : Int = 2
		val nsamps : Int = ((fdata.nrows * 3f)/3f).toInt
		val useGPU : Boolean = true
		val rF = new NewRandomForest(fdata, cats, ntrees, depth, nsamps, useGPU)
		rF
	}

	def prepTreeForTrain2 : NewRandomForest = {
		val x : IMat = load("../Data/digits.mat", "xTrain"); // nfeats * ntrain
		val y : DMat = load("../Data/digits.mat", "yTrain"); // ntrain * 1

		val numCats = 10;
		val n = x.ncols

		val fdata : Mat =  FMat(x) // nfeats x n
		val cats : Mat = sparse(FMat((icol(0->numCats) * iones(1,n) )  == y.t));  //ncats x n
		val ntrees : Int = 4
		val depth : Int = 11
		val nsamps : Int = 512 // ((fdata.nrows * 2f)/3f).toInt
		val useGPU : Boolean = true
		val rF = new NewRandomForest(fdata, cats, ntrees, depth, nsamps, useGPU)
		rF
	}

	/****
	 * TrainingSamples: 60000 samples * 784 feats
 	 * NumTestSamples: 10000 samples * 784 feats
	 * T: 1 D: 11 = 0.87090 (Total Time 205s) on just CPU
	 * T: 4 D: 13 = 0.9325 (Train Time Seconds: 204.421, Test Time Seconds: 0.141) with just sorting on GPU
	 * T: 5 D: 13 = 0.93680
	 * T: 10 D: 13 = 0.95010 (Train Time Seconds: 1153.413) (Test Time Seconds: 0.126)
	 * 
	 * New Version with treepack and sort on GPU: 
	 * T: 4 D: 11: 0.9218 Train Time Seconds: 128.645 Test Time Seconds: 0.203
	 *
	 * New Version with Treepack and sort and minimpurity on GPU (but not completely on GPU)
	 * T: 4 D: 13: 0.9305 Train Time Seconds: 80.015, Test time Seconds: 0.205 Seconds
	 * T: 2 D: 11: 0.8735 Train Time Seconds: 33.303 Test time Seconds: 0.291 Seconds
	 * T: 4 D: 11: 0.9212 Train Time: 61 seconds, Test Time: 13.139 seconds?
	 * T: 4 D: 11: 0.9212 Train Time: 26071 seconds, Test Time: 0.203 Seconds
	 * 
	 * Version with treepack sort on CPU; treepack run twice.
	 * T: 8 D: 11 0.934 Train Time Seconds: 650.379 Test Time Seconds: 0.303
	 *
	 * Version with iterative treepack and sorting on gpu, as well as everything else
	 * T: 8  D: 11 0.9345 Train Time Seconds: 46.983 Test Time Seconds: 0.213
	 * T: 16 D: 11 0.9414 Train Time Seconds: 160.217 Test Time Seconds: 0.263
	 * T: 8 D: 11 nsamps: 512  done prep tree: 2.734 / done train: 50.128 / done test: 0.215 / Accuracy 0.9345/ done calcAccuracy: 0.008
	 ****/
	def testTrain2 : Int = {
		println("start prep tree")
		tic
		val rF = prepTreeForTrain2
		val t0 = toc
		println("done prep tree: " + t0)

		println("start train")
		rF.train
		println("done train: ")

		val xTest : IMat = load("../Data/digits.mat", "xTest"); // nfeats * ntest
		val yTest : DMat = load("../Data/digits.mat", "yTest"); // ntest * 1
		println("start test")
		val yGuess : Mat = rF.classify(FMat(xTest)) 
		println("done test: ")

		// println(yGuess)
		println(calcAccuracy(yGuess, FMat(yTest.t)))
		0
	}

	def prepTreeForTrain3 : NewRandomForest = {
		val r : Runtime = java.lang.Runtime.getRuntime();
		println("freeMemory0: " + r.freeMemory() )
		val x : DMat = load("../Data/8MMNIST/Train7.0/Train7.0.1.mat", "xTrain1"); // nfeats * ntrain
		println("freeMemory0.1: " + r.freeMemory() )
		val y : DMat = load("../Data/8MMNIST/Train7.0/Train7.0.1.mat", "yTrain1"); // 1 * ntrain
		println("freeMemory1: " + r.freeMemory() )

		val numCats = 10;
		val n = x.ncols

		val fdata : Mat =  FMat(x) // nfeats x n
		println("freeMemory2: " + r.freeMemory() )
		val cats : Mat = sparse(FMat((icol(0->numCats) * iones(1,n) )  == y));  //ncats x n
		println("freeMemory3: " + r.freeMemory() )
		val ntrees : Int = 2
		val depth : Int = 11
		val nsamps : Int = 512 // ((fdata.nrows * 2f)/3f).toInt
		val useGPU : Boolean = true
		val rF = new NewRandomForest(fdata, cats, ntrees, depth, nsamps, useGPU)
		rF
	}
	/******
	 *
	 *  Accuracy:0.8120036; T: 2; D: 11; ntrain = 1M; ntest = 1.1M; traintime = 523.675s; testtime = 80.476s
	 *
	 *******/
	def testTrain3 : Int = {
		println("testTrain3")
		println("start prep tree")
		tic
		val rF = prepTreeForTrain3
		val t0 = toc
		println("done prep tree: " + t0)

		println("start train")
		// tic
		rF.train
		// val t1 = toc
		// println("done train: " + t1)

		val xTest : DMat = load("../Data/8MMNIST/Test1.1/Test1.1.mat", "xTest"); // nfeats * ntest
		val yTest : DMat = load("../Data/8MMNIST/Test1.1/Test1.1.mat", "yTest"); // 1 * ntest
		println("start test")
		// tic
		val yGuess : Mat = rF.classify(FMat(xTest)) 
		// val t2 = toc
		println("done test: ")

		// println(yGuess)
		println("calcAccuracy: ")
		// tic
		println(calcAccuracy(yGuess, FMat(yTest)))
		// val t3 = toc
		// println("done calcAccuracy: " + t3)
		0
	}


	def testExtractField : Int = {
		val fieldlengths = 1\1\1\2\2\2
		val cat = 6
		val FieldMaskRShifts = IMat(RandForest.getFieldShifts(fieldlengths))
		val FieldMasks = IMat(RandForest.getFieldMasks(fieldlengths))
		val packedFields = RandForest.packFields(0, 0, 0, 0, cat, 0, fieldlengths)
		val outCat = RandForest.extractField(4, packedFields, FieldMaskRShifts, FieldMasks)
		println("extracted field: " + outCat)
		if (2 == outCat) {
			1
		} else {
			0
		}
	}

	def testExtractAbove : Int = {
		val fieldlengths = 1\1\1\2\2
		val FieldMaskRShifts = IMat(RandForest.getFieldMasks(fieldlengths))
		val packedFields = RandForest.packFields(1, 1, 0, 0, 0, 0, fieldlengths)
		val out = RandForest.extractAbove(2, packedFields, FieldMaskRShifts)
		val eout = 6
		println("extracted field: " + out)
		if (out == eout) {
			1
		} else {
			0
		}
	}


	def testPackFieldsAndExtract : Int = {
		val fieldlengths = 4\4\4\4\4\4
		val packedFields = RandForest.packFields(5, 6, 7, 8, 9, 11, fieldlengths)
		val FieldMaskRShifts = IMat(RandForest.getFieldShifts(fieldlengths))
		val FieldMasks = IMat(RandForest.getFieldMasks(fieldlengths))
		val out0 = RandForest.extractField(0, packedFields, FieldMaskRShifts, FieldMasks)
		val out1 = RandForest.extractField(1, packedFields, FieldMaskRShifts, FieldMasks)
		val out2 = RandForest.extractField(2, packedFields, FieldMaskRShifts, FieldMasks)
		val out3 = RandForest.extractField(3, packedFields, FieldMaskRShifts, FieldMasks)
		val out4 = RandForest.extractField(4, packedFields, FieldMaskRShifts, FieldMasks)
		val out5 = RandForest.extractField(5, packedFields, FieldMaskRShifts, FieldMasks)
		println("extracted fields: " + out0 + " " + out1 + " " + out2 + " " + out3 + " " + out4 + " " + out5 + " ")
		if (out0 == 5 && out1 == 6 && out2 == 7 && out3 == 8 && out4 == 9 && out5 == 11) {
			1
		} else {
			0
		}
	}

	def testSort : Int = {
		val arr = Array(4l,2l,3l,1l)
		val earr = Array(1l,2l,3l,4l)
		RandForest.sortLongs(arr, false)
		println("Result: " + arr.deep.mkString(" "))
		println("Expected: " + earr.deep.mkString(" "))
		if (arr.deep.mkString(" ") == earr.deep.mkString(" ")) {
			1
		} else {
			0
		}
	}

	def testCountC : Int = {
		val arr = Array(1L,1L, 2L, 2L, 2L, 3L, 4L, 5L)
		val ect = 5
		val ct = RandForest.countC(arr)
		println("Result: " + ct)
		println("Expected: " + ect)
		if (ect == ct) {
			1
		} else {
			0
		}
	}

	def testCountCAndMakeC : Int = {
		val arr = Array(1L,1L, 2L, 2L, 2L, 3L, 4L, 5L)
		val ect = 5
		val ct = RandForest.countC(arr)
		println("Result: " + ct)
		println("Expected: " + ect)
		val out = new Array[Long](ect)
		val counts = new Array[Float](ect)
		RandForest.makeC(arr, out, counts)
		val eout = Array(1L,2L,3L,4L,5L)
		val ecounts = Array(2f, 3f, 1f, 1f, 1f)
		println("out: " + out.deep.mkString(" "))
		println("counts: " + counts.deep.mkString(" "))
		println("eout: " + eout.deep.mkString(" "))
		println("ecounts: " + ecounts.deep.mkString(" "))
		if (ect == ct) {
			1
		} else {
			0
		}
	}


	def testMaskAndShifts : Int = {
		val fL = 1\3\1\2\2
		val FieldMaskRShifts = RandForest.getFieldShifts(fL); 
		val FieldMasks = RandForest.getFieldMasks(fL)
		val eFieldMaskRShifts = 8\5\4\2\0
		val eFieldMasks = 1\7\1\3\3
		println("Result Shifts: " + FieldMaskRShifts + " Masks: " + FieldMasks)
		println("Expected Shifts: " + eFieldMaskRShifts + " Masks: " + eFieldMasks)
		if (isAccurate(FieldMaskRShifts, eFieldMaskRShifts) && isAccurate(FieldMasks, eFieldMasks)) {
			1
		} else {
			0
		}
	}

	def testScaleFD : Int = {
		 // def scaleFD(fd : FMat, fb : FMat) 
		 val fd = FMat( 1\2\3 on 4\6\8)
		 val fb = FMat( 1\3 on 4\8)
		 val nifeat = 10
		 val efd = null
		 println("beforeFD: " + fd)
		 val afterFD = RandForest.scaleFD(fd, fb, nifeat)
		 println("afterFD: " + afterFD)
		 0
	}

	def testScaleFDInForest : Int = {
		val rF = prepTreeForTrain2
		0
	}

	// def getFieldShifts(fL : IMat) : IMat = {

	def testRandForestGetFieldShifts : Int = {
		val fL = 2\3\2\1\2\3
		val fieldShifts = RandForest.getFieldShifts(fL)

		val eFieldShifts =  11\8\6\5\3\0
		println("fieldShifts: " + fieldShifts)
		println("eFieldShifts: " + eFieldShifts)
		if (isAccurate(fieldShifts, eFieldShifts)) {
			1
		} else {
			0
		}
	}

	def testFindBoundaries : Int = {
		// findBoundaries(keys:Array[Long], jc:IMat, shift:Int)
		// def findBoundaries(keys:Array[Long], jc:IMat, shift:Int, useGPU : Boolean) = {			
		val x = FMat(1\2\3\4 on 4\7\5\4 on 8\1\6\7)
		val y = IMat(1\0\1\0)
		val numCats = 2;
		val n = x.ncols

		val fdata : Mat =  IMat(x) // nfeats x n
		val cats : Mat = sparse(FMat((icol(0->numCats) * iones(1,n) )  == y));
		(cats) match {
			case (cts : SMat) => {
				println("cts.ir: " + cts.ir.deep.mkString(" "))
				println("cts.jc: " + cts.jc.deep.mkString(" "))
			}
		}
		
		val treenodes = IMat(2\2\3\4)
		val fieldlengths = 1\3\3\3\3\1
		val nsamps = 3
		val outCPU : Array[Long] = RandForest.treePackk(fdata, treenodes, cats, nsamps, fieldlengths, false)
		println("CPU: " + outCPU.deep.mkString(" "))
		// val outGPU : Array[Long] = RandForest.treePackk(fdata,treenodes, cats, 3, fieldlengths, true)
		// println("GPU: " + outGPU.deep.mkString(" "))
		RandForest.sortLongs(outCPU, true)
		val cpuBounds = x.izeros(1, 1 << (14))
		RandForest.findBoundariess(outCPU, cpuBounds, 0, false)
		print(cpuBounds)
		val gpuBounds = x.izeros(1, 1 << (14))
		RandForest.findBoundariess(outCPU, gpuBounds, 0, true)
		print(gpuBounds)

		if (isAccurate(cpuBounds, gpuBounds)) {
			1
		} else {
			0
		}
	}

	 // outg should be an nsamps * nnodes array holding the impurity gain (use maxi2 to get the best)
  // jc should be a zero-based array that points to the start and end of each group of fixed node,jfeat
  // def minImpurity(keys:Array[Long], cnts:IMat, outv:IMat, outf:IMat, outg:FMat, outc:IMat, jc:IMat, fieldlens:IMat, 
  //     ncats:Int, fnum:Int) = {

	def testMinImpurity : Int = {
		val x = FMat(1\2\3\4 on 4\7\5\4 on 8\1\6\7 on 1\2\3\4)
		val y = IMat(3\1\2\3)
		val numCats = 4;
		val n = x.ncols

		val fdata : Mat =  IMat(x) // nfeats x n
		val cats : Mat = sparse(FMat((icol(0->numCats) * iones(1,n) )  == y));
		(cats) match {
			case (cts : SMat) => {
				println("cts.ir: " + cts.ir.deep.mkString(" "))
				println("cts.jc: " + cts.jc.deep.mkString(" "))
			}
		}
		val treenodes = IMat(2\3\3\3)
		val fieldlengths = 1\2\2\2\4\2
		// val ntrees = 2
		// nnodes = 4
		// nsamps = 4
		// nrows = 4
		// nvals = 16
		// val ncats = 4

		val ntrees = 1 << fieldlengths(0)
		val nnodes = 1 << fieldlengths(1)
		val nsamps = 1 << fieldlengths(2)
		val nrows = 1 << fieldlengths(3)
		val nvals = 1 << fieldlengths(4)
		val ncats = 1 << fieldlengths(5)

		val outCPU : Array[Long] = RandForest.treePackk(fdata, treenodes, cats, nsamps, fieldlengths, false)
		println("CPU: " + outCPU.deep.mkString(" "))
		RandForest.sortLongs(outCPU, true)
		val gpuBounds = x.izeros(1, nsamps*ntrees*nnodes+1)
		RandForest.findBoundariess(outCPU, gpuBounds, fieldlengths(5) + fieldlengths(4) + fieldlengths(3), false)
		println("gpuBounds")
		println(gpuBounds)
		val c = RandForest.countC(outCPU)
		val inds = new Array[Long](c)
		val indsCounts = new Array[Float](c)
		RandForest.makeC(outCPU, inds, indsCounts)

		val coutv = IMat(x.izeros(nsamps, ntrees * nnodes)) // TODO: think about number of trees?
		val coutf = IMat(x.izeros(nsamps, ntrees * nnodes))
		val coutg = FMat(x.zeros(nsamps, ntrees * nnodes))
		val coutc = IMat(x.izeros(nsamps, ntrees * nnodes))
		RandForest.minImpurityy(inds, IMat(new FMat(indsCounts.length, 1, indsCounts)), coutv, coutf, coutg, coutc, gpuBounds, fieldlengths, ncats, 0, false)
		val goutv = IMat(x.izeros(nsamps, ntrees * nnodes)) // TODO: think about number of trees?
		val goutf = IMat(x.izeros(nsamps, ntrees * nnodes))
		val goutg = FMat(x.izeros(nsamps, ntrees * nnodes))
		val goutc = IMat(x.izeros(nsamps, ntrees * nnodes))
		RandForest.minImpurityy(inds, IMat(new FMat(indsCounts.length, 1, indsCounts)), goutv, goutf, goutg, goutc, gpuBounds, fieldlengths, ncats, 0, true)
		println("-----------------")
		println("outv")
		isAccurate(goutv, coutv);
		println("outf")
		isAccurate(goutf, coutf);
		println("outg")
		isAccurate(goutg, coutg);
		println("outc")
		isAccurate(goutc, coutc);
		println(maxi2(goutg, 1))
		println("-----------------")
		// if (isAccurate(coutv, goutv) && isAccurate(coutf, goutf) && isAccurate(coutg, goutg) && isAccurate(coutc, goutc)) {
		// 	1
		// } else {
		// 	0
		// }
		0
	}

	// def treePackk(sfdata : Mat, treenodes : Mat, cats : Mat, nsamps : Int, fieldlengths: Mat, useGPU : Boolean) : Array[Long] = {
 //    (sfdata, treenodes, cats, nsamps, fieldlengths, useGPU)
 	def testTreePackk : Int = {
 		// val ITree = 0; val INode = 1; val JFeat = 2; val IFeat = 3; val IVFeat = 4; val ICat = 5
		val fieldlengths = 0\2\1\2\4\1


 		val x = FMat(1\2\3\4 on 4\7\8\9 on 8\1\12\15 on 4\1\2\3)
		val y = IMat(1\0\1\0)
		val numCats = 2;
		val n = x.ncols

		val fdata : Mat =  IMat(x) // nfeats x n
		val cats : Mat = sparse(FMat((icol(0->numCats) * iones(1,n) )  == y));
		(cats) match {
			case (cts : SMat) => {
				println("cts.ir: " + cts.ir.deep.mkString(" "))
				println("cts.jc: " + cts.jc.deep.mkString(" "))
			}
		}
		
		val treenodes = IMat(2\2\3\3)
		// ntree, nodes, nsamps, 
		val nsamps = 1 << fieldlengths(JFeat)
		val outGPU : Array[Long] = RandForest.treePackk(fdata,treenodes, cats, nsamps, fieldlengths, true)
		println("GPU: " + outGPU.deep.mkString(" "))
		val outCPU : Array[Long] = RandForest.treePackk(fdata,treenodes, cats, nsamps, fieldlengths, false)
		println("CPU: " + outCPU.deep.mkString(" "))
	
		val FieldMaskRShifts =  RandForest.getFieldShifts(fieldlengths)
		val FieldMasks =  RandForest.getFieldMasks(fieldlengths)
		var i = 0
		while (i < outGPU.length) {
			println("i: " + i)
			// extractField(fieldNum : Int, packedFields : Long, FieldShifts : IMat, FieldMasks : IMat)
			// val ITree = 0; val INode = 1; val JFeat = 2; val IFeat = 3; val IVFeat = 4; val ICat = 5
			val outGPU0 = RandForest.extractField(0, outGPU(i), FieldMaskRShifts, FieldMasks)
			val outGPU1 = RandForest.extractField(1, outGPU(i), FieldMaskRShifts, FieldMasks)
			val outGPU2 = RandForest.extractField(2, outGPU(i), FieldMaskRShifts, FieldMasks)
			val outGPU3 = RandForest.extractField(3, outGPU(i), FieldMaskRShifts, FieldMasks)
			val outGPU4 = RandForest.extractField(4, outGPU(i), FieldMaskRShifts, FieldMasks)
			val outGPU5 = RandForest.extractField(5, outGPU(i), FieldMaskRShifts, FieldMasks)
			val outCPU0 = RandForest.extractField(0, outCPU(i), FieldMaskRShifts, FieldMasks)
			val outCPU1 = RandForest.extractField(1, outCPU(i), FieldMaskRShifts, FieldMasks)
			val outCPU2 = RandForest.extractField(2, outCPU(i), FieldMaskRShifts, FieldMasks)
			val outCPU3 = RandForest.extractField(3, outCPU(i), FieldMaskRShifts, FieldMasks)
			val outCPU4 = RandForest.extractField(4, outCPU(i), FieldMaskRShifts, FieldMasks)
			val outCPU5 = RandForest.extractField(5, outCPU(i), FieldMaskRShifts, FieldMasks)
			println("GPU: ITree:" + outGPU0 + " INode:" + outGPU1 + " JFeat:" + outGPU2 + " IFeat:" + outGPU3 + " IVFeat:" + outGPU4 + " ICat:" + outGPU5)
			println("CPU: ITree:" + outCPU0 + " INode:" + outCPU1 + " JFeat:" + outCPU2 + " IFeat:" + outCPU3 + " IVFeat:" + outCPU4 + " ICat:" + outCPU5)
			i += 1
		}

		1
 	}

 	// TODO: asdfasfsad
 	def treePackAndSort : Int = {
 		val fieldlengths = 0\2\1\2\4\1
 		val x = FMat(1\2\3\4 on 4\7\8\9 on 8\1\12\15 on 4\1\2\3)
		val y = IMat(1\0\1\0)
		val numCats = 2;
		val n = x.ncols
		val fdata : Mat =  IMat(x) // nfeats x n
		val cats : Mat = sparse(FMat((icol(0->numCats) * iones(1,n) )  == y));
		(cats) match {
			case (cts : SMat) => {
				println("cts.ir: " + cts.ir.deep.mkString(" "))
				println("cts.jc: " + cts.jc.deep.mkString(" "))
			}
		}
		val treenodes = IMat(2\2\3\3)
		// ntree, nodes, nsamps, 
		val nsamps = 1 << fieldlengths(JFeat)
		val outGPU : Array[Long] = RandForest.treePackk(fdata,treenodes, cats, nsamps, fieldlengths, true)
		println("GPU: " + outGPU.deep.mkString(" "))
		0
 	}

	/********************************************************/
	/*********************** HELPERS ************************/
	/********************************************************/
	def calcAccuracy(guess : Mat , actual : Mat) : Float = {
		// println("guess")
		// println(guess)
		// println("actual")
		// println(actual)
		val accuracyThreshold = 10E-7
		var correctness : Mat = null
		(guess) match {
			case (i : IMat) => {
				correctness = (guess == actual)
			}
			case (f : FMat) => {
				correctness = ((guess - actual) < accuracyThreshold)
			}
		}
		// println("correctness")
		// println(correctness)
		val summed1 = sum(correctness, 1)
		val summed2 = sum(summed1, 2)
		return FMat(summed2)(0,0) / (correctness.length.toFloat)
	}	

	def isAccurate(guess : Mat , actual : Mat) : Boolean = {
		val accuracy = calcAccuracy(guess, actual) 
		// println("accuracy: " + accuracy)
		accuracy == 1.0f
	}

}

val t = new testHashNewRandomForest
// t.prepTreeForTrain2
// t.testTrain1
// t.testTrain2
t.testTrain3
// t.prepTreeForTrain3
// t.testRandForestGetFieldShifts
// t.testScaleFD
// t.testScaleFDInForest
// t.testSort
// t.testCountC
// t.testCountCAndMakeC
// t.testExtractField
// t.testExtractAbove
// t.testPackFieldsAndExtract
// t.testMaskAndShifts
// t.testExtractField
// t.testTreePackk
// t.testPackFieldsAndExtract
// t.testFindBoundaries
// t.testMinImpurity
