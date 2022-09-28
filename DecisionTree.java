import java.io.Serializable;
import java.util.ArrayList;
import java.text.*;
import java.lang.Math;

public class DecisionTree implements Serializable {

	DTNode rootDTNode;
	int minSizeDatalist; //minimum number of datapoints that should be present in the dataset so as to initiate a split
	
	// Mention the serialVersionUID explicitly in order to avoid getting errors while deserializing.
	public static final long serialVersionUID = 343L;
	
	public DecisionTree(ArrayList<Datum> datalist , int min) {
		minSizeDatalist = min;
		rootDTNode = (new DTNode()).fillDTNode(datalist);
	}

	class DTNode implements Serializable{
		//Mention the serialVersionUID explicitly in order to avoid getting errors while deserializing.
		public static final long serialVersionUID = 438L;
		boolean leaf;
		int label = -1;      // only defined if node is a leaf
		int attribute; // only defined if node is not a leaf
		double threshold;  // only defined if node is not a leaf

		DTNode left, right; //the left and right child of a particular node. (null if leaf)

		DTNode() {
			leaf = true;
			threshold = Double.MAX_VALUE;
		}

		
		// this method takes in a datalist (ArrayList of type datum). It returns the calling DTNode object 
		// as the root of a decision tree trained using the datapoints present in the datalist variable and minSizeDatalist.
		// Also, KEEP IN MIND that the left and right child of the node correspond to "less than" and "greater than or equal to" threshold
		DTNode fillDTNode(ArrayList<Datum> datalist) {
			int counter = 0;
			double best_avg_entropy = Double.POSITIVE_INFINITY;
			int best_attr = -1;
			double best_threshold = -1;
			for (int c = 0; c < datalist.size(); c++) {
				if (datalist.get(c).y==datalist.get(0).y) {
					counter++;
				}
			}
			if (datalist.size() >= minSizeDatalist) {
				if (counter == datalist.size()) {
					this.leaf = true;
					this.label = datalist.get(0).y;
					this.left = null;
					this.right = null;
					return this;
				} else {
					for (int j = 0; j < datalist.get(0).x.length; j++) {
						for (int i = 0; i < datalist.size(); i++) {
							double cutoff = datalist.get(i).x[j];
							ArrayList<Datum> sublist1 = new ArrayList<Datum>();
							ArrayList<Datum> sublist2 = new ArrayList<Datum>();
							for (int k = 0; k < datalist.size(); k++) {
								if (datalist.get(k).x[j] < cutoff) {
									sublist1.add(datalist.get(k));
								} else {
									sublist2.add(datalist.get(k));
								}
							}
							double sublist1Entropy = calcEntropy(sublist1);
							double sublist2Entropy = calcEntropy(sublist2);
							double w1 = (double) sublist1.size()/datalist.size();
							double w2 = (double) sublist2.size()/datalist.size();
							double current_avg_entropy = (w1*(sublist1Entropy))+(w2*(sublist2Entropy));
							if (best_avg_entropy > current_avg_entropy) {
								best_avg_entropy = current_avg_entropy;
								best_attr = j;
								best_threshold = datalist.get(i).x[j];
							}
						}
					}
					if (best_avg_entropy == calcEntropy(datalist)) {
						this.leaf = true;
						this.label = findMajority(datalist);
						this.left = null;
						this.right = null;
						return this;
					} else {
						this.attribute = best_attr;
						this.threshold = best_threshold;
						this.leaf = false;
						this.left = new DTNode();
						this.right = new DTNode();
						ArrayList<Datum> data1 = new ArrayList<Datum>();
						ArrayList<Datum> data2 = new ArrayList<Datum>();
						for (int m = 0; m < datalist.size(); m++) {
							if (datalist.get(m).x[this.attribute]<this.threshold) {
								data1.add(datalist.get(m));
							} else {
								data2.add(datalist.get(m));
							}
						}
						this.left.fillDTNode(data1);
						this.right.fillDTNode(data2);
						return this;
					}
				}
			} else {
				this.leaf = true;
				this.label = findMajority(datalist);
				this.left = null;
				this.right = null;
				return this;
			}
			
		}



		// This is a helper method. Given a datalist, this method returns the label that has the most
		// occurrences. In case of a tie it returns the label with the smallest value (numerically) involved in the tie.
		int findMajority(ArrayList<Datum> datalist) {
			
			int [] votes = new int[2];

			//loop through the data and count the occurrences of datapoints of each label
			for (Datum data : datalist)
			{
				votes[data.y]+=1;
			}
			
			if (votes[0] >= votes[1])
				return 0;
			else
				return 1;
		}




		// This method takes in a datapoint (excluding the label) in the form of an array of type double (Datum.x) and
		// returns its corresponding label, as determined by the decision tree
		int classifyAtNode(double[] xQuery) {
			if (this.leaf == true) {
				return this.label;
			} else if (xQuery[this.attribute]<this.threshold) {
				return this.left.classifyAtNode(xQuery);
			} else {
				return this.right.classifyAtNode(xQuery);
			}
		}
		


		//given another DTNode object, this method checks if the tree rooted at the calling DTNode is equal to the tree rooted
		//at DTNode object passed as the parameter
		public boolean equals(Object dt2){
			if (dt2 instanceof DTNode) {
				if (((DTNode)dt2) != null && this != null) {
					if (((DTNode)dt2).leaf == true && this.leaf == true && ((DTNode)dt2).label == this.label) {
						return (((DTNode)dt2).label == this.label);
					} else if (((DTNode)dt2).leaf != true && this.leaf != true && ((DTNode)dt2).attribute == this.attribute && ((DTNode)dt2).threshold == this.threshold) {
						return (this.left.equals(((DTNode)dt2).left)&& this.right.equals(((DTNode)dt2).right));
					}
				}
			}
			return false;
		}
	}



	//Given a dataset, this returns the entropy of the dataset
	double calcEntropy(ArrayList<Datum> datalist) {
		double entropy = 0;
		double px = 0;
		float [] counter= new float[2];
		if (datalist.size()==0)
			return 0;
		double num0 = 0.00000001,num1 = 0.000000001;

		//calculates the number of points belonging to each of the labels
		for (Datum d : datalist)
		{
			counter[d.y]+=1;
		}
		//calculates the entropy using the formula specified in the document
		for (int i = 0 ; i< counter.length ; i++)
		{
			if (counter[i]>0)
			{
				px = counter[i]/datalist.size();
				entropy -= (px*Math.log(px)/Math.log(2));
			}
		}

		return entropy;
	}


	// given a datapoint (without the label) calls the DTNode.classifyAtNode() on the rootnode of the calling DecisionTree object
	int classify(double[] xQuery ) {
		return this.rootDTNode.classifyAtNode( xQuery );
	}

	// Checks the performance of a DecisionTree on a dataset
	// This method is provided in case you would like to compare your
	// results with the reference values provided in the PDF in the Data
	// section of the PDF
	String checkPerformance( ArrayList<Datum> datalist) {
		DecimalFormat df = new DecimalFormat("0.000");
		float total = datalist.size();
		float count = 0;

		for (int s = 0 ; s < datalist.size() ; s++) {
			double[] x = datalist.get(s).x;
			int result = datalist.get(s).y;
			if (classify(x) != result) {
				count = count + 1;
			}
		}

		return df.format((count/total));
	}


	//Given two DecisionTree objects, this method checks if both the trees are equal by
	//calling onto the DTNode.equals() method
	public static boolean equals(DecisionTree dt1,  DecisionTree dt2)
	{
		boolean flag = true;
		flag = dt1.rootDTNode.equals(dt2.rootDTNode);
		return flag;
	}

}

