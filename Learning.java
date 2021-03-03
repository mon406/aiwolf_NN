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
	
	private DataSetIterator trainingData;	// �g���[�j���O�f�[�^
	private DataSetIterator testData;		// �e�X�g�f�[�^

	public void start() throws IOException, InterruptedException {

		// �S�f�[�^��ǂݍ���
		File dataFile = new File("data.csv");
		BufferedReader buffer_read = new BufferedReader(new FileReader(dataFile));
		List<String> humanLines = new ArrayList<String>();
		List<String> wolfLines = new ArrayList<String>();
		String line;
		while ((line = buffer_read.readLine()) != null) {
			if (line.startsWith("g")) {	// ��s�ڂ�"gameLog20200814-061252-509\game"�����O����
				System.out.println("Not data: " + line);
			} else if (line.startsWith("0")) {
				humanLines.add(line);
			} else {
				wolfLines.add(line);
			}
		}
		buffer_read.close();
		
		// �w�K�p�ƕ]���p�Ƀ����_����2�������ăt�@�C���ɏ����o��
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
		
		// ���߂Ċw�K�p�f�[�^�ƕ]���p�f�[�^��ǂݍ���
		int batchSize = 32;
		int labelIndex = 0;
		RecordReader recode_reader1 = new CSVRecordReader();
		recode_reader1.initialize(new FileSplit(new File("trainingData.csv")));
		trainingData = new RecordReaderDataSetIterator.Builder(recode_reader1, batchSize).regression(labelIndex).build();
		RecordReader recode_reader2 = new CSVRecordReader();
		recode_reader2.initialize(new FileSplit(new File("testData.csv")));
		testData = new RecordReaderDataSetIterator.Builder(recode_reader2, batchSize).regression(labelIndex).build();
		
		// NN�̍\��
		int numInputs = 11;			// ���͑w�f�q��
		int numHidden1Units = 32;	// ��1�B��w�f�q��
		int numHidden2Units = 16;	// ��2�B��w�f�q��
		double learningRate = 0.001;	// �w�K��
		DenseLayer hiddenLayer1 = new DenseLayer.Builder().nIn(numInputs).nOut(numHidden1Units)
				.activation(Activation.RELU).dropOut(0.5).build(); // ��1�B��w
		DenseLayer hiddenLayer2 = new DenseLayer.Builder().nIn(numHidden1Units).nOut(numHidden2Units)
				.activation(Activation.RELU).dropOut(0.5).build(); // ��2�B��w
		OutputLayer outputLayer = new OutputLayer.Builder(LossFunction.XENT).nIn(numHidden2Units).nOut(1)
				.activation(Activation.SIGMOID).build(); // �o�͑w
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(System.currentTimeMillis()) // �����̎�
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // �œK���A���S���Y���F�m���I�ŋ}�~���@
				.weightInit(WeightInit.VAR_SCALING_NORMAL_FAN_AVG) // �d�݂̏����l�F����0���U1�̐��K�������i���͐��{�o�͐��j�̕��ςŊ������K�E�X���z
				.updater(new Adam(learningRate)) // ADAM(A Method for Stochastic Optimization)
				.list().layer(0, hiddenLayer1).layer(1, hiddenLayer2).layer(2, outputLayer).build();

		// early stopping�A���S���Y���̍\��
		int nEpoch = 100; // �I���G�|�b�N
		int evalInterval = 1; // �]���Ԋu
		int patience = 5; // 10�G�|�b�N���P�Ȃ��ŏI��
		EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
				.scoreCalculator(new DataSetLossCalculator(testData, true)) // �X�R�A�Ƃ��ĕ]���p�f�[�^�ɑ΂��鑹����p����
				.evaluateEveryNEpochs(evalInterval).modelSaver(new LocalFileModelSaver(".")) // ���f�����t�@�C���ɕۑ�����
				.saveLastModel(false) // �ŗǂ̃��f���̂ݕۑ�����(�ŐV�̃��f���͕ۑ����Ȃ�)
				.epochTerminationConditions(new MaxEpochsTerminationCondition(nEpoch),
						new ScoreImprovementEpochTerminationCondition(patience))
				.build();
		EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, conf, trainingData);
		trainer.setListener(new MyListener());

		// �w�K�J�n
		trainer.fit();
	}

	/**
	 * NN�̕]��
	 * 
	 * @param model    NN�̃��f��
	 * @param iterator �]���p�f�[�^�̃C�e���[�^
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
	 * �J�X�^�����X�i�����N���X
	 * 
	 * Copyright (c) 2021 OTSUKI Takashi
	 */
	class MyListener implements EarlyStoppingListener<MultiLayerNetwork> {

		public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration<MultiLayerNetwork> esConfig,
				MultiLayerNetwork net) {
			// �������Ȃ�
		}

		public void onCompletion(EarlyStoppingResult<MultiLayerNetwork> esResult) {
			System.out.println("�I���̗��R: " + esResult.getTerminationReason());
			System.out.println("�I���̏ڍ�: " + esResult.getTerminationDetails());
			System.out.println("���G�|�b�N��: " + esResult.getTotalEpochs());
			System.out.println("�ŗǃG�|�b�N: " + esResult.getBestModelEpoch());
			System.out.println("�ŗǃG�|�b�N�ɂ�����X�R�A: " + esResult.getBestModelScore());

			startEvaluation(esResult.getBestModel(), testData);
		}

		public void onStart(EarlyStoppingConfiguration<MultiLayerNetwork> esConfig, MultiLayerNetwork net) {
			// �������Ȃ�
		}
	}
	
}