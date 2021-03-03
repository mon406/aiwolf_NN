/**
 * Learn.java
 *
 * @author momma
 */
package jp.ac.yamagata_u.st.momma.aiwolf;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.EvaluationBinary;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Learning {

	public static void main(String[] args) {
		try {
			new Learning().start();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private DataSetIterator trainingData;	// トレーニングデータ
	private DataSetIterator testData;		// テストデータ

	public void start() throws IOException, InterruptedException {

		// 全データを読み込み
		File dataFile = new File("data.csv");
		BufferedReader buffer_read = new BufferedReader(new FileReader(dataFile));
		List<String> humanLines = new ArrayList<String>();
		List<String> wolfLines = new ArrayList<String>();
		String line;
		while ((line = buffer_read.readLine()) != null) {
			if (line.startsWith("g")) {	// 一行目の"gameLog20200814-061252-509\game"を除外する
				System.out.println("Not data: " + line);
			} else if (line.startsWith("0")) {
				humanLines.add(line);
			} else {
				wolfLines.add(line);
			}
		}
		buffer_read.close();
		
		// 学習用と評価用にランダムで2等分してファイルに書き出す
		int DataNum = humanLines.size() < wolfLines.size() ? humanLines.size() : wolfLines.size();
		Collections.shuffle(humanLines);
		Collections.shuffle(wolfLines);
		List<String> allLines = new ArrayList<String>();
		allLines.addAll(humanLines.subList(0, DataNum));
		allLines.addAll(wolfLines.subList(0, DataNum));
		Collections.shuffle(allLines);
		
		BufferedWriter buffer_writer1 = new BufferedWriter(new FileWriter("trainingData.csv"));
		for (int i = 0; i < DataNum; i++) {
			buffer_writer1.write(allLines.get(i));
			buffer_writer1.newLine();
		}
		buffer_writer1.close();
		BufferedWriter buffer_writer2 = new BufferedWriter(new FileWriter("testData.csv"));
		for (int i = DataNum; i < allLines.size(); i++) {
			buffer_writer2.write(allLines.get(i));
			buffer_writer2.newLine();
		}
		buffer_writer2.close();
		
		// 改めて学習用データと評価用データを読み込む
		int batchSize = 32;
		int labelIndex = 0;
		RecordReader recode_reader1 = new CSVRecordReader();
		recode_reader1.initialize(new FileSplit(new File("trainingData.csv")));
		trainingData = new RecordReaderDataSetIterator.Builder(recode_reader1, batchSize).regression(labelIndex).build();
		RecordReader recode_reader2 = new CSVRecordReader();
		recode_reader2.initialize(new FileSplit(new File("testData.csv")));
		testData = new RecordReaderDataSetIterator.Builder(recode_reader2, batchSize).regression(labelIndex).build();
		
		// NNの構成
		int numInputs = 11;			// 入力層素子数
		int numHidden1Units = 32;	// 第1隠れ層素子数
		int numHidden2Units = 16;	// 第2隠れ層素子数
		double learningRate = 0.001;	// 学習率
		DenseLayer hiddenLayer1 = new DenseLayer.Builder().nIn(numInputs).nOut(numHidden1Units)
				.activation(Activation.RELU).dropOut(0.5).build(); // 第1隠れ層
		DenseLayer hiddenLayer2 = new DenseLayer.Builder().nIn(numHidden1Units).nOut(numHidden2Units)
				.activation(Activation.RELU).dropOut(0.5).build(); // 第2隠れ層
		OutputLayer outputLayer = new OutputLayer.Builder(LossFunction.XENT).nIn(numHidden2Units).nOut(1)
				.activation(Activation.SIGMOID).build(); // 出力層
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(System.currentTimeMillis()) // 乱数の種
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // 最適化アルゴリズム：確率的最急降下法
				.weightInit(WeightInit.VAR_SCALING_NORMAL_FAN_AVG) // 重みの初期値：平均0分散1の正規乱数を（入力数＋出力数）の平均で割ったガウス分布
				.updater(new Adam(learningRate)) // ADAM(A Method for Stochastic Optimization)
				.list().layer(0, hiddenLayer1).layer(1, hiddenLayer2).layer(2, outputLayer).build();

		// early stoppingアルゴリズムの構成
		int nEpoch = 100; // 終了エポック
		int evalInterval = 1; // 評価間隔
		int patience = 5; // 10エポック改善なしで終了
		EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
				.scoreCalculator(new DataSetLossCalculator(testData, true)) // スコアとして評価用データに対する損失を用いる
				.evaluateEveryNEpochs(evalInterval).modelSaver(new LocalFileModelSaver(".")) // モデルをファイルに保存する
				.saveLastModel(false) // 最良のモデルのみ保存する(最新のモデルは保存しない)
				.epochTerminationConditions(new MaxEpochsTerminationCondition(nEpoch),
						new ScoreImprovementEpochTerminationCondition(patience))
				.build();
		EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, conf, trainingData);
		trainer.setListener(new MyListener());

		// 学習開始
		trainer.fit();
	}

	/**
	 * NNの評価
	 * 
	 * @param model    NNのモデル
	 * @param iterator 評価用データのイテレータ
	 * 
	 * Copyright (c) 2021 OTSUKI Takashi
	 */
	void startEvaluation(MultiLayerNetwork model, DataSetIterator iterator) {
		EvaluationBinary eval = new EvaluationBinary();
		iterator.reset();
		while (iterator.hasNext()) {
			DataSet dataSet = iterator.next();
			INDArray features = dataSet.getFeatures();
			INDArray lables = dataSet.getLabels();
			INDArray predicted = model.output(features, false);
			eval.eval(lables, predicted);
		}
		System.out.println(eval.stats());
	}

	/**
	 * カスタムリスナ内部クラス
	 * 
	 * Copyright (c) 2021 OTSUKI Takashi
	 */
	class MyListener implements EarlyStoppingListener<MultiLayerNetwork> {

		public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration<MultiLayerNetwork> esConfig,
				MultiLayerNetwork net) {
			// 何もしない
		}

		public void onCompletion(EarlyStoppingResult<MultiLayerNetwork> esResult) {
			System.out.println("終了の理由: " + esResult.getTerminationReason());
			System.out.println("終了の詳細: " + esResult.getTerminationDetails());
			System.out.println("総エポック数: " + esResult.getTotalEpochs());
			System.out.println("最良エポック: " + esResult.getBestModelEpoch());
			System.out.println("最良エポックにおけるスコア: " + esResult.getBestModelScore());

			startEvaluation(esResult.getBestModel(), testData);
		}

		public void onStart(EarlyStoppingConfiguration<MultiLayerNetwork> esConfig, MultiLayerNetwork net) {
			// 何もしない
		}
	}
	
}