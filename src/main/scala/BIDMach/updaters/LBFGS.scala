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
  var myS:FMat = null
  var myY:FMat = null
  var mydW:Array[Mat] = null
  var myW:Mat = null
//  var myYY:Array[Mat] = null
  var myLBFGSindex = 0
  var myLBFGSiold  = 0
  var vk:Mat = null
  var eye:Mat = null 
  var rho:FMat = 1.0f 
  var rk:Mat = null 
  var Hk:Mat = null 
  var myGrad:Mat = null
  var myYY:Array[FMat] = null
  var mySS:Array[FMat] = null
  var nvec = 1
  var upThld = 1.0
  var myIndex =0
  
  override def init(model0:Model) = {
    model = model0
    modelmats = model.modelmats;
    updatemats = model.updatemats;
//    myYY = model.myYY;  //TODO:fix it
    val mm = modelmats(0);
    mask = opts.mask;
    val nmats = modelmats.length;
    nvec = updatemats(0).nrows;
  //   sumSq = new Array[Mat](nmats);
  //  myS  = new Array[Mat](nmats);
//    myY  = new Array[Mat](nmats);
 //   mydW  = model.mydW;
    myYY  = new Array[FMat](nvec);
    mySS  = new Array[FMat](nvec);
    mydW  = new Array[Mat](nvec);
    println(updatemats(0).nrows, updatemats(0).ncols);
    vk = zeros(modelmats(0).ncols, modelmats(0).ncols);
    rk = zeros(modelmats(0).ncols, modelmats(0).ncols);
    Hk = zeros(modelmats(0).ncols, modelmats(0).ncols);
    eye = zeros(modelmats(0).ncols, modelmats(0).ncols);
    
    myGrad   = zeros(updatemats(0).ncols, 1) + 0.0;
    myS  =  zeros(updatemats(0).ncols, 5) + 0.1f;
    myY  =  zeros(updatemats(0).ncols, 5) + 0.1f;
    myW = modelmats(0).t + 0.001;
    for(i<-0 until modelmats(0).ncols){eye(i,i)=1.0f;}
      
    for (i<-0 until nvec) {
 //     sumSq(i) = modelmats(i).ones(modelmats(i).nrows, modelmats(i).ncols) *@ opts.initsumsq
 //     myS  =  modelmats(i).ones(modelmats(i).nrows, modelmats(i).ncols) *@ opts.initmys
 //     myY  =  updatemats(i).ones(updatemats(i).nrows, updatemats(i).ncols) *@ opts.initmys
 //     mydW(i) = modelmats(i).ones(modelmats(i).nrows, modelmats(i).ncols) *@ opts.initmys
      mydW(i)  =  zeros(updatemats(0).ncols, 2); 
      mySS(i)  =  zeros(updatemats(0).ncols, 5) + 0.1;  
      myYY(i)  =  zeros(updatemats(0).ncols, 5) + 0.1;
    
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
   
    var mymats = modelmats(0);    
    var nmats = modelmats.length;
    var mlen = modelmats(0).ncols;
    var tmp1 = updatemats(0).t; 
    var myq = zeros(updatemats(0).ncols, 6);
    var myro = zeros(5,1);
    var myal = zeros(5,1);
    var mybe = zeros(5,1);
    var myr = zeros(updatemats(0).ncols,6); 
    var k=0;
    var maxIter = 40;
    
  val nsteps = if (step == 0) 1f else {
      if (firstStep == 0f) {
        firstStep = step
        1f
      } else {
        step / firstStep
      }
    }
    stepn.set(1f/nsteps);
   
    var nnmats = 0;
    if (opts.lrate.ncols > 1) {
        lrate <-- opts.lrate(?,nnmats);
      } else {
        lrate <-- opts.lrate;
      }
          
    tmp1 = -( ( updatemats(nnmats) *@ (lrate *@ (stepn ^ te)) ) ).t;
    //tmp1 = updatemats(nnmats);
    nvec = updatemats(nnmats).nrows;
  //  println(nmats,nvec);
     
   if( (myIndex < 6) || ( (myIndex < maxIter) && upThld > 0.00001 ) )
   {
    for (i <- 0 until nvec) {
//     var mm = modelmats(i);
//     var um = updatemats(i);
 //      myY = ( - updatemats(i) - myYY(i) ).asInstanceOf[FMat];
//   mydW(i) = mydW(i) - updatemats(i);
//       myS = mydW(i).asInstanceOf[FMat]; //updatemats(i);
         myY = myYY(i);
         myS = mySS(i);
      //   println("ipass",ipass);
     //  if (opts.waitsteps < nsteps) 
    //  {
        
        var grad = tmp1(?,i); 
        var gradold = mydW(0);
        myGrad = grad + (-gradold(?,i));
        var tmp2 = 0.0 ; 
        for(j <-0 until myY.ncols)
        {
         
         tmp2 = ( myY(?,j).t * myS(?,j) ).data(0);   
        // println(tmp);
         if(tmp2 != 0.0){ myro(j,0) = 1.0 / tmp2; }
         else{myro(j,0) = 0.0;}
        }//for(j)
        // println("myro:",myro);
        myq(?,5)=gradold(?,i);
        for(j <-0 until myY.ncols)
        {
          k=4-j; //println("k=",k);
          myal(k,0) = myro(k,0) * ( myS(?,k).t * myq(?,(k+1)) ).data(0);
          tmp2 = myal(k,0); 
          myq(?,(k))=myq(?,(k+1) )-( tmp2 * myY(?,k) );
           
         // println(tmp,myq(?,(k)));
        }
        myr(?,0)=(0.01*eye)*myq(?,0);
     //   var tmp2=norm( myY(?,0) ^* myq(?,0) );
     //   println(tmp2);
        for(j <-0 until myY.ncols)
        {
          mybe(j,0) = myro(j,0)* (myY(?,j).t * myr(?,j) ).data(0) ;
          tmp2 = (myal(j,0)-mybe(j,0));
          myr(?,(j+1)) = myr(?,j)+ ( (myal(j,0)-mybe(j,0)) * myS(?,j) );
      //    println(tmp,myr(?,(j+1)));
        }
     //   println(mybe,myal);
    //    println(myr.nrows,myr.ncols);
        for(j <-0 until 4)
        {
         myY(?,j)= myY(?,(j+1) ); 
         myS(?,j)= myS(?,(j+1) );
        }      
          myY(?,4) = myGrad;
     //   if(myro(4,0) != 0){ myS(?,4) = ( (-0.1) * myr(?,5) ); }
     //   else
         { 
          myS(?,4) = ( (-0.1)*grad );    
         }
      //  println("grad:",grad);
      //  println("r:",myr(?,5));

        //mydW(i) = myS;  //to modify
        mySS(i) = myS;
        myYY(i) = myY;
        mymats(i,?) = ((myS(?,4)).t);
        myW(?,i) = myW(?,i) + (myS(?,4));
       //  myYY(i) ~ (-modelmats(i)) + 0f;   //history d_updatemats
        
    //    if(ipass>0){
    //      myY(i) <-- (newupdate - mydW(i));
    //    } else{
    //      myY(i) <-- newupdate;
    //    }  
        
       // if (mask != null) modelmats(i) ~ modelmats(i) *@ mask;
        var tmp3 = ( ( mymats(0,?).t )/mymats.ncols ); 
        upThld = norm( tmp3 ); 
      //  println("upThld",upThld);
      }//for i
     //   println( mydW(i) );
     //  mydW(i) ~ (-updatemats(i)) *@ one ; //history updatemats   
    
    } // if( (myIndex < 6) || ( (myIndex < maxIter) && upThld > 0.00001 ) )
    
       // if(upThld != 0.0){ upThld = norm( (mymats(0,?).t) / (myW(?,0)) );}  
       
        modelmats(0) = myW.t;      
       // modelmats(nnmats)  ~  modelmats(nnmats)  + ( mymats );
        if (mask != null)   modelmats(nnmats)  ~  modelmats(nnmats)  *@ mask;
        myIndex = myIndex + 1;
        mydW(0)  = tmp1;
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

