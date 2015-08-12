package BIDMach.updaters


/**
 * @author Anna
 */

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models._
import org.scalatest._;
import org.scalatest.junit._;
import org.scalatest.prop._;
import org.junit.runner.RunWith

@RunWith(classOf[JUnitRunner])
class COCOATest extends FunSuite with BeforeAndAfter {  // with Checkers ???
  before{
    //create global matrix
  }
  
  test("test for gradient") {  
//    assert(false)
      val x = DMat(2,2);  
      /* input x_data is one row for one sample;
       * eg:x_data = List(1.0,-1.0, 1.0,-1.0),
       * after arraycopy, 
       * x = +------------+
       *     |  1.0,  1.0 |
       *     | -1.0, -1.0 |
       *     +------------+ 
       * this input are 2 samples (1.0,1.0),(-1.0,-1.0);
       */
      val modelmats = List(1.0,-1.0, 1.0,-1.0).toArray;
      val x_data = x.data;
      System.arraycopy(modelmats, 0, x.data, 0, 4);
      val y = DMat(1,2)
      val yvalues = List(1.0,-1.0).toArray; 
      System.arraycopy(yvalues, 0, y.data, 0, 2);
      val lambda = 0.1;
      val w = DMat(2,1);
      val ww = List(1.0,1.0).toArray;
      System.arraycopy(ww, 0, w.data, 0, 2);
      val z = (y*(x*w)-1.0)/(lambda*2.0);
      val z_0_0 = z(0,0);
   //   println(w);
    assert(z_0_0 == 15.0);
    
  }
  
    test("test for update") {  
//    assert(false)
      val x = DMat(2,2);  
      /* input x_data is one row for one sample;
       * eg:x_data = List(1.0,-1.0, 1.0,-1.0),
       * after arraycopy, 
       * x = +------------+
       *     |  1.0,  1.0 |
       *     | -1.0, -1.0 |
       *     +------------+ 
       * this input are 2 samples (1.0,1.0),(-1.0,-1.0);
       */
      val modelmats = List(1.0,-1.0, 1.0,-1.0).toArray;
      val x_data = x.data;
      System.arraycopy(modelmats, 0, x.data, 0, 4);
      val y = DMat(1,2)
      val yvalues = List(1.0,-1.0).toArray; 
      System.arraycopy(yvalues, 0, y.data, 0, 2);
      val lambda = 0.1;
      val w = DMat(2,1);
      val ww = List(1.0,1.0).toArray;
      System.arraycopy(ww, 0, w.data, 0, 2);
      val myS = DMat(2,2);
      
      val myY = DMat(2,2);
      val grad = DMat(2,1);
      val B = DMat(2,2);
      val eta0 = 1;
      val p = DMat(2,1)
      val pvalues = List(2.0,1.0).toArray; 
      System.arraycopy(pvalues, 0, p.data, 0, 2); 
      val alpha = DMat(2,1);
      val alphavalues = List(3.0,1.0).toArray; 
      System.arraycopy(alphavalues, 0, alpha.data, 0, 2);
      val alphaNew = alpha - eta0 * p;
 
      val up = x*(y*(alphaNew - alpha )).t/(lambda*2.0);
//      println(up);   
      val wNew = w + up;
//      println(wNew);
      assert(wNew(0,0) == -4.0);     

    }
  
  after {
    //destroy global matrix
  }
}
