package BIDMach.updaters

/**
 * @author Anna
 */

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._


class COCOA(override val opts:COCOA.Opts = new COCOA.Options) extends Updater {
  
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
  var myYY:Array[FMat] = null
  var mySS:Array[FMat] = null
  var myCOCOAindex = 0
  var myCOCOAiold  = 0
  var vk:Mat = null
  var eye:Mat = null 
  var rho:FMat = 1.0f 
  var rk:Mat = null 
  var Hk:Mat = null 
  var inX:Mat = null
  var inY:Mat = null
  var myW:Mat = null
  var myAlpha:Mat = null
  var newAlpha:Mat = null
  var myGrad:Mat = null
  var maxIter = 40
  var upThld = 1.0
  var myThld = 1.0;
  
  override def init(model0:Model) = {
   
    upThld = 1.0;
    
    model = model0
 
    modelmats = model.modelmats;
    updatemats = model.updatemats;
    inX = model.gmats(0);
    inX = inX / (norm(inX));
    inY = model.gmats(1).t;
 //   println(inX);
 //   myYY = model.myYY;  //TODO:fix it
    val mm = modelmats(0);
    mask = opts.mask;
    val nmats = modelmats.length;
  //   sumSq = new Array[Mat](nmats);
  //  myS  = new Array[Mat](nmats);
//    myY  = new Array[Mat](nmats);
    mydW  = new Array[Mat](1);
    myYY  = new Array[FMat](inY.ncols);
    mySS  = new Array[FMat](inY.ncols);
    
    myAlpha   = zeros(inX.ncols, 2) + 0.001;
    newAlpha   = zeros(inX.ncols, 2) + 0.0;
    myW = modelmats(0).t + 0.001;//zeros(inX.nrows,2)+ 0.0;
    myGrad   = zeros(inX.ncols, 1) + 0.0;
     
    vk = zeros(inX.ncols, inX.ncols);
    rk = zeros(inX.ncols, inX.ncols);
    Hk = zeros(inX.ncols, inX.ncols);
    eye = zeros(inX.ncols, inX.ncols);
    for(i<-0 until inX.ncols){eye(i,i)=1.0f;}
    
    myS  =  zeros(inX.ncols, 5) + 0.1f;
    myY  =  zeros(inX.ncols, 5) + 0.1f;
      
    for (i <- 0 until inY.ncols) {
    mySS(i)  =  zeros(inX.ncols, 5) + 0.1;  
    myYY(i)  =  zeros(inX.ncols, 5) + 0.1;
    }
 //   for (i <- 0 until nmats) {
 //     sumSq(i) = modelmats(i).ones(modelmats(i).nrows, modelmats(i).ncols) *@ opts.initsumsq
    mydW(0) = zeros(inX.ncols, 2);
 //     mydW(i)  =  ones(inX.ncols, 1) *@ opts.initmys
 //     mydW(0) = modelmats(0).ones(modelmats(0).nrows, modelmats(0).ncols) *@ opts.initmys;
 //       mydW(i) = mydW(i) *@ opts.initmys 
     // ww = model.datamats;   
//    }
    stepn = mm.zeros(1,1);
    one = mm.ones(1,1);
    ve = mm.zeros(opts.vexp.nrows, opts.vexp.ncols);
    te = mm.zeros(opts.texp.nrows, opts.texp.ncols);
    lrate = mm.zeros(opts.lrate.nrows, 1);
    ve <-- opts.vexp;
    te <-- opts.texp;
    //myS = mm.zeros(modelmats(0).nrows, modelmats(0).ncols) + opts.initmys;
    //myY = mm.zeros(modelmats(0).nrows, modelmats(0).ncols) + opts.initmys;
    
    myCOCOAindex = 0;
    myCOCOAiold  = 0;
    
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
    
    var myq = zeros(inX.ncols, 6);
    var myro = zeros(5,1);
    var myal = zeros(5,1);
    var mybe = zeros(5,1);
    var myr = zeros(inX.ncols,6);
    val nsteps = if (step == 0) 1f else {
      if (firstStep == 0f) {
        firstStep = step
        1f
      } else {
        step / firstStep
      }
    }
    stepn.set(1f/nsteps);
    
    var k = 0;
    var nmats = modelmats.length;
    var mlen = modelmats(0).ncols;
    var dW = modelmats(0);
 //   var tmp2 = inX.t * myW;
 //   println(tmp2.nrows,tmp2.ncols);
 //   println(inY.nrows,inY.ncols);
    var xx=inX(?,0).t;
    var yy=inY(0,?);
 //   println(inX.nrows,inX.ncols,myW.nrows,myW.ncols);
    var tmp1=zeros(inX.ncols,2);
    var myupdate = 0.0; 
    
  if( (myCOCOAindex < 6) || ( (myCOCOAindex < maxIter) && upThld > 10.0 ) )
  {
   for(i <-0 until inY.ncols)
   {
    //tmp1(?,i) = (inY(?,i) - (inX.t*myW(?,i)) ) / (inX.ncols);
    tmp1(?,i)=(( (inY(?,i) dotr (inX.t * myW(?,i)) )-1.0 )*(0.0001*inX.ncols) ) / (norm(inX)); 
 //   myupdate = ( norm( (inY(?,i) dotr (inX.t * myW(?,i)) )-1.0 ) )/(inX.ncols);
 //   myThld = myupdate - upThld;
//    println(tmp1.nrows,tmp1.ncols,myupdate.nrows,myupdate.ncols,myupdate);
 //   var grad = (( (inY(?,i) dotr (inX.t * myW(?,i)) ) -1.0)*(0.1*inX.ncols) )/ norm(inX); 
    var grad = tmp1(?,i); //(( (inY dotr (inX.t * myW) ) -1.0)*(0.1*inX.ncols) )/ norm(inX); 
 //   println(myThld);
//    println("grad:", grad);
    var gradold = mydW(0);
    myY = myYY(i);
    myS = mySS(i);
    
    myGrad ~ grad + (-gradold(?,i));
   // println(myW);
     
        var tmp = 0.0 ; 
        for(j <-0 until myY.ncols)
        {
         
         tmp = ( myY(?,j).t * myS(?,j) ).data(0);   
        // println(tmp);
         if(tmp != 0.0){ myro(j,0) = 1.0 / tmp; }
         else{myro(j,0) = 0.0;}
        }//for(j)
        // println("myro:",myro);
        myq(?,5)=gradold(?,i);
        for(j <-0 until myY.ncols)
        {
          k=4-j; //println("k=",k);
          myal(k,0) = myro(k,0) * ( myS(?,k).t * myq(?,(k+1)) ).data(0);
          tmp = myal(k,0); 
          myq(?,(k))=myq(?,(k+1) )-( tmp * myY(?,k) );
           
         // println(tmp,myq(?,(k)));
        }
        myr(?,0)=(0.01*eye)*myq(?,0);
     //   var tmp2=norm( myY(?,0) ^* myq(?,0) );
     //   println(tmp2);
        for(j <-0 until myY.ncols)
        {
          mybe(j,0) = myro(j,0)* (myY(?,j).t * myr(?,j) ).data(0) ;
          tmp = (myal(j,0)-mybe(j,0));
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
        if(myro(4,0) != 0){myS(?,4) = (-0.1)*myr(?,5);}
        else{ 
          myS(?,4) = (-0.9)*grad;    
         }
     
       //   println("grad:",grad);
       //   println("r:",myr(?,5));
       //   println(myS);
       //   println(myY);
     //   var tmp = ( pk );
     //   println(myS);
   //     for(k <-0 until 1000){
        newAlpha(?,i) = myAlpha(?,i) + (myS(?,4));
     //   for(j<-0 until newAlpha.nrows){
     //   newAlpha(j,?) = min(max((newAlpha(j,?)), 0.0), 1.0);
     //   }
     //   myS(?,4) = newAlpha(?,i) + (- myAlpha(?,i) );
        
     //   println(newAlpha(?,i));
        myAlpha(?,i) = newAlpha(?,i);
        
      //  if (mask != null) myAlpha ~ myAlpha *@ mask;          

 //       println(inX.nrows,inX.ncols);
 //       println(myS.nrows,myS.ncols);
 //       var tmp2 = inY(?,0) dotr (myS(?,4));
 //       println(tmp2.nrows,tmp2.ncols);
        
      //  for(j <-0 until inY.ncols)
      //  {
            var tmp2 = ((inX * (inY(?,i) dotr (myS(?,4)) ) / (0.0001 * inX.ncols) ) );
             upThld = norm( tmp2/myW(?,i) );
            myW(?,i) = myW(?,i) + (inX * (inY(?,i) dotr (myS(?,4)) ) / (0.0001 * inX.ncols) ) ;
         // myW(?,i) = myW(?,i) + ( (inX * myS(?,4) ) / (0.1 * inX.ncols) ) ;
          
         //   println(upThld);
        //  myW(?,i) = myW(?,i) / norm(myW(?,i));
        //  println(myW);
          mySS(i) = myS;
          myYY(i) = myY;
       // }
   }    //for(i <-0 until inY.ncols)
       
        modelmats(0) = myW.t;         
      //  modelmats(i) ~ modelmats(i) + (myS);      
        if (mask != null) modelmats(0) ~ modelmats(0) *@ mask;
        
    //    println(myW.nrows,myW.ncols);
    //    println(myW);
    //    println(modelmats(i).nrows,modelmats(i).ncols);
        myCOCOAindex = myCOCOAindex + 1;
        mydW(0)  = tmp1;
      //  upThld = myupdate;
     }//if(myCOCOAindex < maxIter)
  
     } //update

  
} //class LBFGS updater


object COCOA {
  trait Opts extends Grad.Opts {
    var vexp:FMat = 0.5f
    var epsilon = 1e-5f
    var initsumsq = 1e-5f
    var initmys = 1e-1f
  }
  
  class Options extends Opts {}
}

