package BIDMach.models

import BIDMat.{SBMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Sorting._
import edu.berkeley.bid.CUMAT
import BIDMach.models.RForest._
import BIDMach.models.RandForest._
import jcuda._
import jcuda.runtime._
import jcuda.runtime.JCuda._

/**********
 * fdata = nfeats x n 
 * fbounds = nfeats x 2
 * treenodes = ntrees x n
 * cats = ncats x n // labels 1 or greater
 * ntrees = ntrees
 * nsamps = nsamps
 * fieldLengths = 1 x 5 or 5 x 1
 * depth = depth 
 **********/
class NewRandomForest(fdata : Mat, cats : Mat, ntrees : Int, depth : Int, nsamps : Int, useGPU : Boolean) {
	// val ITree = 0; val INode = 1; val IRFeat = 2; val IVFeat = 3; val ICat = 4
	val ITree = 0; val INode = 1; val JFeat = 2; val IFeat = 3; val IVFeat = 4; val ICat = 5

	var fbounds : Mat = mini(fdata, 2) \ maxi(fdata, 2)
	var fieldLengths : Mat = fdata.iones(1, 6)
	val itree = (Math.log(ntrees)/ Math.log(2)).toInt; val inode = depth; 
	val jfeat = (Math.log(nsamps)/ Math.log(2)).toInt
	val ifeat = (Math.log(fdata.nrows)/ Math.log(2)).toInt + 1; // TODO? errr.... not sure about this.....
	val ncats = cats.nrows
	var icat : Int = (Math.log(ncats)/ Math.log(2)).toInt + 1 // todo fix mat element access

	val ivfeat = Math.min(10,  64 - itree - inode - jfeat - ifeat - icat); 
	fieldLengths <-- (itree\inode\jfeat\ifeat\ivfeat\icat) 
	val n = fdata.ncols
	val treenodes = fdata.izeros(ntrees, fdata.ncols)
	val nnodes = (math.pow(2, depth).toInt)
	val treesMetaInt = fdata.izeros(4, (ntrees * nnodes)) // irfeat, threshold, cat, isLeaf
	treesMetaInt(2, 0->treesMetaInt.ncols) = (ncats) * iones(1, treesMetaInt.ncols)
	treesMetaInt(3, 0->treesMetaInt.ncols) = (-1) * iones(1, treesMetaInt.ncols)
	val treesMetaInt2 = fdata.izeros(4, (ntrees * nnodes)) // irfeat, threshold, cat, isLeaf
	treesMetaInt2(2, 0->treesMetaInt2.ncols) = (ncats) * iones(1, treesMetaInt2.ncols)
	treesMetaInt2(3, 0->treesMetaInt2.ncols) = (-1) * iones(1, treesMetaInt2.ncols)

	val treesData = fdata.izeros(1, ntrees * nnodes)
	
	// var FieldMaskRShifts : Mat = null;  var FieldMasks : Mat = null
	var sFData : Mat = null
	(fieldLengths, fdata, fbounds) match {
		case (fL : IMat, fd : FMat, fb : FMat) => {
			// FieldMaskRShifts = RForest.getFieldMaskRShifts(fL); FieldMasks = RForest.getFieldMasks(fL)
			println("fd prescale")
			println(fd)
			sFData = RandForest.scaleFD(fd, fb, math.pow(2, fL(IVFeat)).toInt - 1)
			println("fd postscale")
			println(sFData)
		}
	}
	
	def train {
		var totalTrainTime = 0f
		(sFData, treenodes, cats, nsamps, fieldLengths, treesMetaInt, depth) match {
			case (sfd : IMat, tn : IMat, cts : SMat, nsps : Int, fL : IMat, tMI : IMat, d : Int) => {
				var d = 0
				while (d <  depth) {
					println("d: " + d)
					val jc : IMat = null
					// treePackk(sfdata : Mat, treenodes : Mat, cats : Mat, nsamps : Int, fieldlengths: Mat, useGPU : Boolean)
					val treePacked : Array[Long] = RandForest.treePackk(sfd, tn, cts, nsps, fL, useGPU) // useGPU
					RandForest.sortLongs(treePacked, true)
					val c = RandForest.countC(treePacked)
					val inds = new Array[Long](c)
					val indsCounts = new Array[Float](c)
					RandForest.makeC(treePacked, inds, indsCounts)
					// RandForest.updateTreeData(treePacked, fL, ncats, tMI, depth, d == (depth - 1), RandForest.getFieldShifts(fL), RandForest.getFieldMasks(fL))
					// println("tMI correct")
					// println(tMI)
					// def findBoundaries(keys:Array[Long], jc:IMat, shift:Int)
					
					val jccc = sfd.izeros(1, nnodes * ntrees * nsamps + 1)
					RandForest.findBoundariess(inds, jccc, RandForest.getFieldShifts(fL)(JFeat), true)
					
					val outv = IMat(sfd.izeros(nsamps, ntrees * nnodes))
					val outf = IMat(sfd.izeros(nsamps, ntrees * nnodes))
					val outg = FMat(sfd.zeros(nsamps, ntrees * nnodes))
					val outc = IMat(sfd.izeros(nsamps, ntrees * nnodes))
					RandForest.minImpurityy(inds, IMat(new FMat(indsCounts.length, 1, indsCounts)), outv, outf, outg, outc, jccc, fL, ncats, 0, false)
					
					val goutv = IMat(sfd.izeros(nsamps, ntrees * nnodes))
					val goutf = IMat(sfd.izeros(nsamps, ntrees * nnodes))
					val goutg = FMat(sfd.zeros(nsamps, ntrees * nnodes))
					val goutc = IMat(sfd.izeros(nsamps, ntrees * nnodes))
					RandForest.minImpurityy(inds, IMat(new FMat(indsCounts.length, 1, indsCounts)), goutv, goutf, goutg, goutc, jccc, fL, ncats, 0, true)

					println("outv")
					isAccurate(goutv, outv);
					println("outf")
					isAccurate(goutf, outf);
					println("outg")
					isAccurate(goutg, outg);
					println("outc")
					isAccurate(goutc, goutc);

					(treesMetaInt2) match {
						case (tMI2 : IMat) => {
							RandForest.updateTreeDataa(outv, outf, outg, outc, tMI2, fL)
							// isAccurate(tMI2, tMI)
							if (!(d == (depth - 1))) {
								RandForest.treeSteps(tn , sfd, fL, tMI2, depth, ncats, false)
							}
						}
					}
					d += 1
				}
			}
		}
	}


	// returns 1 * n
	def classify(tfdata : Mat) : Mat = {
		var totalClassifyTime : Float = 0f
		val treenodecats = tfdata.izeros(ntrees, tfdata.ncols)
		(tfdata, fbounds, treenodecats, fieldLengths, treesMetaInt2, depth, ncats) match {
			case (tfd : FMat, fb : FMat, tnc : IMat, fL : IMat, tMI2 : IMat, depth : Int, ncts : Int) => {
				val stfd = RandForest.scaleFD(tfd, fb, math.pow(2, fL(IVFeat)).toInt - 1)
				tic; RandForest.treeSearch(tnc, stfd, fL, tMI2, depth, ncts); val t1 = toc
				println("TreeSearch Time: " + t1 + ", Num Elements: " + stfd.length.toFloat + " , Num Bytes: " + Sizeof.INT*stfd.length)
			}
		}
		val out = RandForest.voteForBestCategoriesAcrossTrees(treenodecats.t, ncats) // ntrees * n
		out
	}

	/********************************************************/
	/*********************** HELPERS ************************/
	/********************************************************/
	def calcAccuracy(guess : Mat , actual : Mat) : Float = {
		println("guess")
		println(guess)
		println("actual")
		println(actual)
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
		println("correctness")
		println(correctness)
		val summed1 = sum(correctness, 1)
		val summed2 = sum(summed1, 2)
		return FMat(summed2)(0,0) / (correctness.length.toFloat)
	}	

	def isAccurate(guess : Mat , actual : Mat) : Boolean = {
		val accuracy = calcAccuracy(guess, actual) 
		println("accuracy: " + accuracy)
		accuracy == 1.0f
	}


}