package BIDMach.updaters

/**
 * @author Anna
 */

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._


class LBFGS(override val opts:LBFGS.Opts = new LBFGS.Options) extends Updater {
  
  var firstStep = 0f
  var modelmats:Array[Mat] = null
  var updatemats:Array[Mat] = null  
//  var sumSq:Array[Mat] = null 
  var stepn:Mat = null
  var mask:Mat = null
  var ve:Mat = null
  var te:Mat = null
  var lrate:Mat = null
  var one:Mat = null
  var myS:Mat = null
  var myY:Mat = null
  var mydW:Mat = null
  var myLBFGSindex = 0
  var myLBFGSiold  = 0
  
  override def init(model0:Model) = {
    model = model0
    modelmats = model.modelmats;
    updatemats = model.updatemats;
    val mm = modelmats(0);
    mask = opts.mask;
    val nmats = modelmats.length;
  //   sumSq = new Array[Mat](nmats);
  //  myS  = new Array[Mat](nmats);
  //  myY  = new Array[Mat](nmats);
  //  mydW  = new Array[Mat](nmats);
    
    for (i <- 0 until nmats) {
 //     sumSq(i) = modelmats(i).ones(modelmats(i).nrows, modelmats(i).ncols) *@ opts.initsumsq
      myS  =modelmats(i).ones(modelmats(i).nrows, modelmats(i).ncols) *@ opts.initmys
      myY  =updatemats(i).ones(updatemats(i).nrows, updatemats(i).ncols) *@ opts.initmys
      mydW =zeros(updatemats(i).nrows, updatemats(i).ncols) *@ opts.initmys 
    }
    stepn = mm.zeros(1,1);
    one = mm.ones(1,1);
    ve = mm.zeros(opts.vexp.nrows, opts.vexp.ncols);
    te = mm.zeros(opts.texp.nrows, opts.texp.ncols);
    lrate = mm.zeros(opts.lrate.nrows, 1);
    ve <-- opts.vexp;
    te <-- opts.texp;
    //myS = mm.zeros(modelmats(0).nrows, modelmats(0).ncols) + opts.initmys;
    //myY = mm.zeros(modelmats(0).nrows, modelmats(0).ncols) + opts.initmys;
    
    myLBFGSindex = 1;
    myLBFGSiold  = 0;
    
  } //init
  
  def update2(ipass:Int, step:Long):Unit = {
    modelmats = model.modelmats;
    updatemats = model.updatemats;
    val nsteps = if (step == 0) 1f else {
      if (firstStep == 0f) {
        firstStep = step;
        1f;
      } else {
        step / firstStep;
      }
    }
    stepn.set(nsteps+1);
    val nw = one / stepn;
    val nmats = math.min(modelmats.length, updatemats.length)
    //  println("u2 sumsq %g" format mini(sumSq(0)).dv)
    for (i <- 0 until nmats) {
      val um = updatemats(i);
      val mm = modelmats(i);
//      val ss = sumSq(i);
      if (opts.lrate.ncols > 1) {
        lrate <-- opts.lrate(?,i);
      } else {
        lrate <-- opts.lrate;
      }
      val newsquares = um *@ um;
      newsquares ~ newsquares *@ nw;
 //     ss  ~ ss *@ (one - nw);
 //     ss ~ ss + newsquares;
      if (opts.waitsteps < nsteps) {
 //       val tmp = ss ^ ve;
 //       if (java.lang.Double.isNaN(sum(sum(tmp)).dv)) throw new RuntimeException("ADA0 1 "+i);
 //       tmp ~ tmp *@ (stepn ^ te);
 //       if (java.lang.Double.isNaN(sum(sum(tmp)).dv)) throw new RuntimeException("ADA0 2 "+i);
 //       tmp ~ tmp + opts.epsilon;
 //       mm ~ mm + ((um / tmp) *@ lrate);
        if (java.lang.Double.isNaN(sum(sum(mm)).dv)) throw new RuntimeException("ADA0 3 "+i);
        if (mask != null) mm ~ mm *@ mask;
      }
    }
  }
  
  def update(ipass:Int, step:Long):Unit = { 
  val nsteps = if (step == 0) 1f else {
      if (firstStep == 0f) {
        firstStep = step
        1f
      } else {
        step / firstStep
      }
    }
    stepn.set(1f/nsteps);
    val nmats = modelmats.length;
    val mlen = modelmats(0).ncols;
    val pk  = zeros(modelmats(0).nrows, modelmats(0).ncols);
    val eye = zeros(modelmats(0).ncols, modelmats(0).ncols);
    
 //   var Hk  = zeros(modelmats(0).ncols, modelmats(0).ncols);
   
    for(i<-0 until mlen){eye(i,i)=1.0f;}
    
  //  var vk:FMat = eye;
  //  var Hk = eye;
    //  println("u2 sumsq %g" format mini(sumSq(0)).dv)
    for (i <- 0 until nmats) {
      var mm = modelmats(i);
      var um = updatemats(i);
    //  var yk1 = myY;
    //  var sk1 = myS;
      
      
   //   println("ipass",ipass);
   //   println("i",i);
  //    println(yk1);
  //    println(sk1);
      if (opts.lrate.ncols > 1) {
        lrate <-- opts.lrate(?,i);
      } else {
        lrate <-- opts.lrate;
      }
      if (opts.waitsteps < nsteps) {
        var tmp1 = updatemats(i) *@ (lrate *@ (stepn ^ te));
        
        //var rho = 1f; //yk ^* sk; 
       // var eye =zeros(um.ncols, um.ncols);
    //    var pk = zeros(yk1.ncols,yk1.ncols);
        
        for(j <-0 until myY.nrows){
   //     var yk2 = (yk1(j,?));
   //     var sk2 = (sk1(j,?));
   //     var yk = (myY(j,?)).t;
   //     var sk = (myS(j,?)).t;
   //     println(yk);
          
        val rho = (1/(norm((myY(j,?)).t ^* (myS(j,?)).t) ));
        val max1 = ( (rho)>0f) ;
        if(max1){
        val vk = ( eye - (((myY(j,?)).t *^ (myS(j,?)).t)*(rho) ) );
        val rk = ((myS(j,?)).t *^ (myS(j,?)).t)*(rho);
        val Hk = ( (vk ^* vk) + rk);
      //  println(Hk);
      //  var pk1 = (tmp1(j,?) * Hk);
         myS(j,?) = (tmp1(j,?) * Hk);
       // println(pk1,pk);
        }else{
         myS(j,?) = tmp1(j,?); 
        }//else
        }//for(j)
       // println(tmp1);
     //   var tmp = ( pk );
    //    println(pk);
        modelmats(i) ~ modelmats(i) + myS;
        if (mask != null) modelmats(i) ~ modelmats(i) *@ mask;
    //    println(modelmats(i));
    //    var newupdate = -updatemats(i);
    //    var dw = mydW(i);
    //    myS <-- pk;
    //    println("tmp",tmp);
        
    //    if(ipass>0){
    //      myY(i) <-- (newupdate - mydW(i));
    //    } else{
    //      myY(i) <-- newupdate;
    //    }
       myY <-- ( -updatemats(i) - mydW ); 
 //      println(myY(i));
        
       // if (mask != null) modelmats(i) ~ modelmats(i) *@ mask;
      }//if(opts.waitsteps)
     //   println( mydW(i) );
       mydW <-- -updatemats(i);
 //       println( mydW(i) );
    }//for i
  } //update
} //class LBFGS updater


object LBFGS {
  trait Opts extends Grad.Opts {
    var vexp:FMat = 0.5f
    var epsilon = 1e-5f
    var initsumsq = 1e-5f
    var initmys = 1e-1f
  }
  
  class Options extends Opts {}
}

