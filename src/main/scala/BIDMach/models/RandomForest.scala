package BIDMach.models

import BIDMat.{SBMat,CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Sorting._
import edu.berkeley.bid.CUMAT
import BIDMach.models.RForest._
import jcuda._

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
class RandomForest(fdata : Mat, cats : Mat, ntrees : Int, depth : Int, nsamps : Int, useGPU : Boolean) {
	val ITree = 0; val INode = 1; val IRFeat = 2; val IVFeat = 3; val ICat = 4

	var fbounds : Mat = mini(fdata, 2) \ maxi(fdata, 2)
	var fieldLengths : Mat = fdata.iones(1, 5)
	// def packFields(itree:Int, inode:Int, irfeat:Int, ivfeat:Int, icat:Int, fieldlengths:IMat):Long
	val itree = (Math.log(ntrees)/ Math.log(2)).toInt + 1; val inode = depth + 1; 
	val irfeat = (Math.log(fdata.nrows)/ Math.log(2)).toInt + 1; // TODO? errr.... not sure about this.....
	val ncats = cats.nrows
	var icat : Int = (Math.log(ncats)/ Math.log(2)).toInt + 1 // todo fix mat element access

	val ivfeat = Math.min(10,  64 - itree - inode - irfeat - icat); 
	fieldLengths <-- (itree\inode\irfeat\ivfeat\icat) 
	val n = fdata.ncols
	val treenodes = fdata.izeros(ntrees, fdata.ncols)
	val treesMetaInt = fdata.izeros(4, (ntrees * (math.pow(2, depth).toInt - 1))) // irfeat, threshold, cat, isLeaf
	treesMetaInt(2, 0->treesMetaInt.ncols) = (ncats) * iones(1, treesMetaInt.ncols)
	treesMetaInt(3, 0->treesMetaInt.ncols) = (-1) * iones(1, treesMetaInt.ncols)

	// Shifts and then masks
	var FieldMaskRShifts : Mat = null;  var FieldMasks : Mat = null
	(fieldLengths) match {
		case (fL : IMat) => {
			FieldMaskRShifts = RForest.getFieldMaskRShifts(fL); FieldMasks = RForest.getFieldMasks(fL)
		}
	}
	
	def train {
		// def treePack(fdata:FMat, fbounds:FMat, treenodes:IMat, cats:SMat, nsamps:Int, fieldLengths:IMat)
		var totalTrainTime = 0f
		(fdata, fbounds, treenodes, cats, nsamps, fieldLengths, treesMetaInt, depth, FieldMaskRShifts, FieldMasks) match {
			case (fd : FMat, fb : FMat, tn : IMat, cts : SMat, nsps : Int, fL : IMat, tMI : IMat, d : Int, fMRS : IMat, fM : IMat) => {
				var d = 0
				while (d <  depth) {
					println("d: " + d)
					flip
					val treePacked : Array[Long] = RForest.treePack(fd, fb, tn, cts, nsps, fL)
					val (flop1, time1) = gflop
					totalTrainTime+=time1
					println("treePacked: " + time1)
					
					println("treePacked.length: " + treePacked.length)
					println("treePacked bytes: " + treePacked.length * Sizeof.LONG)
					flip
					RForest.sortLongs(treePacked, useGPU)
					val (flop2, time2) = gflop
					println("sortLongs GPU: FLOPS: " + flop2 + " TIME: " + time2)
					totalTrainTime+=time2

					flip
					RForest.sortLongs(treePacked, !useGPU)
					val (flop5, time5) = gflop
					println("sortLongs CPU: FLOPS: " + flop5 + " TIME: " + time5)


					flip
					RForest.updateTreeData(treePacked, fL, ncats, tMI, depth, d == (depth - 1), fMRS, fM)
					val (flop3, time3) = gflop
					println("updateTreeData: " + time3)
					totalTrainTime+=time3
					if (!(d == (depth - 1))) {
						flip
						RForest.treeSteps(tn , fd, fb, fL, tMI, depth, ncats, false)
						val (flop4, time4) = gflop
						println("treeSteps: " + time4)
						totalTrainTime+=time4
					}
					d += 1
				}
			}
		}
		println("TotalTrainTime: " + totalTrainTime)
	}

	// returns 1 * n
	def classify(tfdata : Mat) : Mat = {
		var totalClassifyTime : Float = 0f
		val treenodecats = tfdata.izeros(ntrees, tfdata.ncols)
		flip
		(tfdata, fbounds, treenodecats, fieldLengths, treesMetaInt, depth, ncats) match {
			case (tfd : FMat, fb : FMat, tnc : IMat, fL : IMat, tMI : IMat, depth : Int, ncts : Int) => {
				RForest.treeSearch(tnc, tfd, fb, fL, tMI, depth, ncts)
			}
		}
		val (flop1, time1) = gflop
		println("treeSearch: " + time1)
		totalClassifyTime+=time1
		println(treenodecats.t)
		flip
		val out = RForest.voteForBestCategoriesAcrossTrees(treenodecats.t, ncats) // ntrees * n
		val (flop2, time2) = gflop
		println("voteForBestCategoriesAcrossTrees: " + time2)
		totalClassifyTime+=time2
		println(treenodecats.t)	
		println("totalClassifyTime: " + totalClassifyTime)	
		out
	}


}