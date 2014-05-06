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
	val itree = (Math.log(ntrees)/ Math.log(2)).toInt + 1; val inode = depth + 1; 
	val jfeat = (Math.log(nsamps)/ Math.log(2)).toInt + 1
	val ifeat = (Math.log(fdata.nrows)/ Math.log(2)).toInt + 1; // TODO? errr.... not sure about this.....
	val ncats = cats.nrows
	var icat : Int = (Math.log(ncats)/ Math.log(2)).toInt + 1 // todo fix mat element access

	val ivfeat = Math.min(10,  64 - itree - inode - jfeat - ifeat - icat); 
	fieldLengths <-- (itree\inode\jfeat\ifeat\ivfeat\icat) 
	val n = fdata.ncols
	val treenodes = fdata.izeros(ntrees, fdata.ncols)
	val treesMetaInt = fdata.izeros(4, (ntrees * (math.pow(2, depth).toInt - 1))) // irfeat, threshold, cat, isLeaf
	treesMetaInt(2, 0->treesMetaInt.ncols) = (ncats) * iones(1, treesMetaInt.ncols)
	treesMetaInt(3, 0->treesMetaInt.ncols) = (-1) * iones(1, treesMetaInt.ncols)

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
					val treePacked : Array[Long] = RandForest.treePackk(sfd, tn, cts, nsps, fL, false) // useGPU
					RandForest.sortLongs(treePacked, true)
					val c = RandForest.countC(treePacked)
					val inds = new Array[Long](c)
					val indsCounts = new Array[Float](c)
					RandForest.makeC(treePacked, inds, indsCounts)

					// def findBoundaries(keys:Array[Long], jc:IMat, shift:Int)
					val temp = fL(ITree) + fL(INode) + fL(JFeat) 
					println("fL(ITree) + fL(INode) + fL(JFeat): " + temp)
					val jccc = sfd.izeros(1, 1 << (fL(ITree) + fL(INode) + fL(JFeat)))
					RandForest.findBoundariess(inds, jccc, RandForest.getFieldShifts(fL)(JFeat), true)
					// println(jccc)
					// def minImpurity(keys:Array[Long], cnts:IMat, outv:IMat, outf:IMat, outg:FMat, jc:IMat, fieldlens:IMat, 
     // 					ncats:Int, fnum:Int)
					RandForest.updateTreeData(treePacked, fL, ncats, tMI, depth, d == (depth - 1), RandForest.getFieldShifts(fL), RandForest.getFieldMasks(fL))
					if (!(d == (depth - 1))) {
						RandForest.treeSteps(tn , sfd, fL, tMI, depth, ncats, false)
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
		(tfdata, fbounds, treenodecats, fieldLengths, treesMetaInt, depth, ncats) match {
			case (tfd : FMat, fb : FMat, tnc : IMat, fL : IMat, tMI : IMat, depth : Int, ncts : Int) => {
				val stfd = RandForest.scaleFD(tfd, fb, math.pow(2, fL(IVFeat)).toInt - 1)
				RandForest.treeSearch(tnc, stfd, fL, tMI, depth, ncts)
			}
		}
		val out = RandForest.voteForBestCategoriesAcrossTrees(treenodecats.t, ncats) // ntrees * n
		out
	}


}