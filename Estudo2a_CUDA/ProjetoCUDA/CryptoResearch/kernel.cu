#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <string.h>
#include <map>
#include <iostream>
#include <stdint.h>

#include <stdlib.h>
#include <fstream>
#include <iomanip> 
#include <sstream>
#include <climits> 
#include <random>
#include <algorithm>
#include <vector>
#include <time.h>

#include <numeric>
#include <cmath>
#include <set>
#include <bitset>
#include <ctime>
#include <chrono>
#include <cstdint>
#include <cstring>   

#include "sha256.cuh"

#define MYGPUID 0 //usa a GPU 0.
#define PPC 67108864 //nº total de chaves candidatas a gerar

//Atalhos para tipos inteiros de 8/16/32/64 bits.
#define U8 unsigned char 
#define U16 uint16_t
#define U32 unsigned int
#define U64 unsigned long long 

//===================================================================================================
//Estruturas
struct ReportBase {
	U32 found; //...............................................flag true means final found
	U32 final; //...............................................index of final solution
};


//Geradores de valores aleatorios
std::random_device rd; //........................................Dispositivo sistema - fonte de entropia
std::mt19937 gen(rd()); //.......................................Gerador baseado na fonte de entropia 
std::uniform_int_distribution<U32> dist(0, UINT32_MAX); //.......distribuição generica para um U32 32 bits
std::uniform_real_distribution<double> dist01(0.0, 1.0); //......distribuição uniforme para sorteios com distribuições personalizadas

//De uso geral
ReportBase host_report; //.......................................Reporte para teste de finalizacao lado host

//Apontadores para reserva de ram GPU
U32* _devPK = NULL; //...........................................chaves privadas

//Determinação de objetivo (HASH)
__device__ U32 hobj[8]; //.......................................hash objetivo
// U64: [31:0] = índice da chave, [63:32] = melhor semelhança (bits)
__device__ U64 maxsim; //........................................máxima semelhança observada + chave que a deu
// hash (SHA256) da chave que deu origem ao bestsim — preenchido por computeBestHash()
__device__ U32 _besthash[8]; //.................................hash da melhor chave reportada

//Chaves candidatas
__constant__ U32* _PK[1]; //[536870912]; ........................pks (U32[8] - size = 8*512*512*256 = 536870912 U32 = 2 147 483 648 bytes +- 2.15 GB) 

//Estrutura de reporte para indicar se solução foi encontrada
__device__ ReportBase _reportBase; //............................Contadores / Stats

//Estrutura geradora do set de chaves candidatas //  Total de bits para indexar: 5×3 + 2×4 = 15+8 = 23 bits de índice por iteração.
__constant__ U32 _MSTATE_0[8] = { 0x8E28A2B7, 0xBEDFE793, 0x84472248, 0x1907CA58, 0x46F25AF2, 0x6E1AEEAA, 0x8B634598, 0xC725E9A1 };
__constant__ U32 _MSTATE_1[8] = { 0x8AE99B0F, 0x9C616FCD, 0x98946F12, 0x01B69816, 0xA79C7177, 0x635089CB, 0xC706AA55, 0x09B2607F };
__constant__ U32 _MSTATE_2[8] = { 0x6802C3F4, 0x952E3CF4, 0x682E340B, 0xA8C174F8, 0xEFBA8938, 0xAB9B02FC, 0x0892CFBB, 0x9C431B61 };
__constant__ U32 _MSTATE_3[8] = { 0x37E902F8, 0xFEC9E949, 0x275F020F, 0xC44F020F, 0x5663E99E, 0x25797690, 0xC5C763BE, 0xB935DF11 };
__constant__ U32 _MSTATE_4[8] = { 0xA92BFDCC, 0xADCEDFE6, 0x53C402D6, 0x16170B90, 0x188EAB34, 0x5FE9834F, 0x01597CE0, 0x57559352 };
__constant__ U32 _MSTATE_5[8] = { 0x492C6505, 0x75D56469, 0xC8089FEB, 0xC23EB3EB, 0xBCC0EDFC, 0xA2A94DF7, 0x69E5DF35, 0x40F6484C };
__constant__ U32 _MSTATE_6[16] = { 0xAEC981DC, 0x43E13C2B, 0x53363C23, 0x3CE94BE7, 0x2D7540F9, 0xF2968AD2, 0x32A2A8CA, 0x4F39B113, 0xD4BEE854, 0x7CF46880, 0xFB592DBE, 0xAC5133FE, 0x9060C8A0, 0xBEE2728E, 0x4EA220BB, 0xEA379D86 };
__constant__ U32 _MSTATE_7[16] = { 0xDACA5118, 0x0675CC9E, 0x0675EE10, 0x398AEC90, 0xBE17D16C, 0xCDE8B183, 0x71AC9317, 0x59B4AFC9, 0x2B57E1B5, 0x704558CB, 0xB353E5AD, 0x6D0643EB, 0x020CFA42, 0x4E8EE0E6, 0xF5E3E423, 0x1B066747 };


//======================================================

//Calculo usado por rotina ECC
__device__ void computePK(U32 pkidx, U32* v) { 
	U32* p = _PK[0] + (pkidx << 3);
	v[0] = p[0];
	v[1] = p[1];
	v[2] = p[2];
	v[3] = p[3];
	v[4] = p[4];
	v[5] = p[5];
	v[6] = p[6];
	v[7] = p[7];
}

//Copia a chave pkidx na gpu para v lado host
//<< 3 = ×8 porque cada chave ocupa 8 U32 (32 bytes = 512 bits) na memória
void h_computePK(U32 pkidx, U32* v) { cudaMemcpy(v, _devPK + (pkidx << 3), 32, cudaMemcpyDeviceToHost); }

//Gera a partir de MSTATE e SHA256 e preenche _cK[] e _PK[]
__global__ void setKeys() {
	//<< 9 = ×512 = blockDim.x, i.e. threadId = blockIdx.x*512 + threadIdx.x
	int threadId = (blockIdx.x << 9) | threadIdx.x;

	U32 localIdx;
	U32 v[8], vh[8];
	U32 IDX;
	U32* p_PK = NULL;
	U32 sLocalIdx;

	for (U32 i = 0; i < 256; i++) {

		localIdx = (i << 18) | threadId; // 256 iterações × 512 threads
		sLocalIdx = localIdx << 3; // <=> x8 (8 U32 por chave)
		IDX = localIdx;

		p_PK = _PK[0] + sLocalIdx;

		v[0] = _MSTATE_0[IDX & 0x07]; IDX >>= 3;
		v[1] = _MSTATE_1[IDX & 0x07]; IDX >>= 3;
		v[2] = _MSTATE_2[IDX & 0x07]; IDX >>= 3;
		v[3] = _MSTATE_3[IDX & 0x07]; IDX >>= 3;
		v[4] = _MSTATE_4[IDX & 0x07]; IDX >>= 3;
		v[5] = _MSTATE_5[IDX & 0x07]; IDX >>= 3;
		v[6] = _MSTATE_6[IDX & 0x0F]; IDX >>= 4;
		v[7] = _MSTATE_7[IDX & 0x0F];

		computeSHA256(v, vh);

		p_PK[0] = vh[0];
		p_PK[1] = vh[1];
		p_PK[2] = vh[2];
		p_PK[3] = vh[3];
		p_PK[4] = vh[4];
		p_PK[5] = vh[5];
		p_PK[6] = vh[6];
		p_PK[7] = vh[7];
	}
}

//Constrói uma máscara de 256 bits (8 U32) com EXATAMENTE n bits a 1,
//em posições aleatórias: embaralha as 256 posições [0..255] e escolhe as primeiras n.
void fillRandomBits(int n, U32* V) {

	// 1. Zerar os 8 elementos (256 bits)
	for (int i = 0; i < 8; ++i)
		V[i] = 0;

	// 2. Gerar lista de posições [0..255]
	std::vector<int> positions(256);
	for (int i = 0; i < 256; ++i)
		positions[i] = i;

	// 3. Embaralhar e escolher 90
	std::shuffle(positions.begin(), positions.end(), gen);

	// 4. Marcar os 90 bits escolhidos
	for (int i = 0; i < n; ++i) {
		int bitIndex = positions[i];
		int wordIndex = bitIndex / 32;       // índice em V[0..7]
		int bitPos = bitIndex % 32;          // bit dentro de V[wordIndex]
		V[wordIndex] |= (1u << bitPos);
	}
}

//Gera a partir de _PK[] e m[]
__global__ void updateKeys(U32 m0, U32 m1, U32 m2, U32 m3, U32 m4, U32 m5, U32 m6, U32 m7) {
	int threadId = (blockIdx.x << 9) | threadIdx.x;
	if (threadId == 0) { maxsim = 0; } //reset do bestsim (U64) para a nova ronda

	U32 sLocalIdx;
	U32 vh[8];

	U32* p_PK = NULL;

	for (U32 i = 0; i < 256; i++) {

		sLocalIdx = ((i << 18) | threadId) << 3;

		p_PK = _PK[0] + sLocalIdx;

		vh[0] = p_PK[0] ^ m0;
		vh[1] = p_PK[1] ^ m1;
		vh[2] = p_PK[2] ^ m2;
		vh[3] = p_PK[3] ^ m3;
		vh[4] = p_PK[4] ^ m4;
		vh[5] = p_PK[5] ^ m5;
		vh[6] = p_PK[6] ^ m6;
		vh[7] = p_PK[7] ^ m7;

		p_PK[0] = vh[0];
		p_PK[1] = vh[1];
		p_PK[2] = vh[2];
		p_PK[3] = vh[3];
		p_PK[4] = vh[4];
		p_PK[5] = vh[5];
		p_PK[6] = vh[6];
		p_PK[7] = vh[7];
	}
}




//===================================================================================================
// Calculo SHA256: verify()

__global__ void computeAllHash() {
	int threadId = (blockIdx.x << 9) | threadIdx.x;
	U32 hash[8];
	U32 index;
	U32 sim, localmaxsim = 0;
	U32 localBestIdx = 0; //índice da chave que deu o melhor sim desta thread
	U32* p_PK = NULL;

	for (U32 i = 0; i < 256; i++) {

		// índice da chave = (i << 18) | threadId (0 .. PPC-1)
		// o offset na memória é índice × 8 U32, como em setKeys/updateKeys
		index = (i << 18) | threadId;
		p_PK = _PK[0] + (index << 3);

		computeSHA256(p_PK, hash);

		// Compute number of corresponding bits
		sim = 256 - (
			__popc(hash[0] ^ hobj[0])
			+ __popc(hash[1] ^ hobj[1])
			+ __popc(hash[2] ^ hobj[2])
			+ __popc(hash[3] ^ hobj[3])
			+ __popc(hash[4] ^ hobj[4])
			+ __popc(hash[5] ^ hobj[5])
			+ __popc(hash[6] ^ hobj[6])
			+ __popc(hash[7] ^ hobj[7])
			);


		if (sim > localmaxsim) { localmaxsim = sim; localBestIdx = index; }

		// Se encontrou a chave final
		if (sim == 256u) {
			atomicExch(&_reportBase.found, 1u);
			atomicExch(&_reportBase.final, index);
		}
	}

	//Empacota (semelhança << 32) | índice e faz atomicMax global:
	//ganha a maior semelhança (em empate, o maior índice)
	atomicMax(&maxsim, ((U64)localmaxsim << 32) | (U64)localBestIdx);
}

//calcula o SHA256 da chave (índice) que deu origem ao bestsim e fica guardado em _besthash[].
//Usa a MESMA função device computeSHA256() da rotina de verificação => hash idêntico ao que foi testado na GPU.
__global__ void computeBestHash(U32 idx) {
	U32* p_PK = _PK[0] + (idx << 3); //<< 3 = ×8 U32 por chave (32 bytes)
	computeSHA256(p_PK, _besthash);
}

void verify() {

	// Transferir dados para controlo de fim
	cudaMemcpyFromSymbol(&host_report, _reportBase, sizeof(ReportBase), 0, cudaMemcpyDeviceToHost);

	// verifica se objetivo atingido
	if (host_report.found == 1) {

		//conclusão: ===========================================================================================
		U32 finalKeyIdx = host_report.final;
		U32 finalKey[8];
		h_computePK(finalKeyIdx, finalKey);

		// Apresenta resultado
		printf("\n************** FINALIZADO ! ******************************");
		printf("\n0x%.8X,0x%.8X,0x%.8X,0x%.8X,0x%.8X,0x%.8X,0x%.8X,0x%.8X", finalKey[0], finalKey[1], finalKey[2], finalKey[3], finalKey[4], finalKey[5], finalKey[6], finalKey[7]);
		printf("\n************** FINALIZADO ! ******************************");
		printf("\n\n*** DONE ! ***\n");

		system("pause");
		while (true); //Programa para aqui =====================================================
	}
}
U32 globalBestSim = 0;
U32 globalBestIdx = 0;   //índice da chave que deu o melhor bestsim (global, para o relatório final)
U32 globalBestKey[8] = { 0 }; //cópia host da melhor chave — para o relatório final
U32 chavesReportadas = 0; //nº de chaves (best) reportadas — paragem obrigatória a 32
auto inicio = std::chrono::steady_clock::now();
void getBestReport() { //********** VISUALIZAÇÃO DA CHAVE QUE DEU ORIGEM AO BESTSIM ********************
	//                              ===================================================================

	U64 h_maxsim;

	cudaMemcpyFromSymbol(&h_maxsim, maxsim, sizeof(U64), 0, cudaMemcpyDeviceToHost);

	//desempacotar: [63:32] = semelhança, [31:0] = índice da chave
	U32 h_sim = (U32)(h_maxsim >> 32);
	U32 h_idx = (U32)(h_maxsim);

	if (h_sim > globalBestSim) {

		auto agora = std::chrono::steady_clock::now();

		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(agora - inicio);

		globalBestSim = h_sim;
		globalBestIdx = h_idx; //guarda o índice da melhor chave (global)
		double percent = ((1.0) * globalBestSim / 256.0) * 100;
		printf("\nmax sim => %u (%.2f%%) ", h_sim, percent);

		std::cout << " @: " << ms.count() << " ms";

		// A chave candidata que deu origem ao bestsim
		U32 bestKey[8];
		h_computePK(h_idx, bestKey);
		for (int i = 0; i < 8; i++) globalBestKey[i] = bestKey[i]; //guarda para o relatório final
		printf("\nchave   => 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X",
			bestKey[0], bestKey[1], bestKey[2], bestKey[3],
			bestKey[4], bestKey[5], bestKey[6], bestKey[7]);

		//nº de bits DIFERENTES em relação ao objetivo — é exatamente a soma de __popc:
		//256 - sim
		printf("\ndif bits => %u / 256 (bits diferentes vs hash objetivo)", (unsigned)(256 - h_sim));

		//apresenta o HASH (SHA256) desta chave — calculado na GPU com a mesma computeSHA256()
		//usada em computeAllHash(), para comparar lado a lado com o hash objetivo
		computeBestHash << <1, 1 >> > (h_idx);
		cudaDeviceSynchronize();
		U32 bestHash[8];
		cudaMemcpyFromSymbol(bestHash, _besthash, 32, 0, cudaMemcpyDeviceToHost);
		printf("\nhash    => 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X",
			bestHash[0], bestHash[1], bestHash[2], bestHash[3],
			bestHash[4], bestHash[5], bestHash[6], bestHash[7]);
	}

}


//===================================================================================================

void init() {

	// INITIALISE DEVICE 
	cudaSetDevice(MYGPUID);                             //..........Seleciona a GPU para reniciar 
	cudaDeviceReset();                                  //..........Reset apenas essa GPU
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync); //..........Define comportamento de espera
	cudaSetDevice(MYGPUID);                             //..........Recria o contexto limpo com as flags
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);    //..........Define cache L1 preferido

	// Allocate memory on device 
	cudaMalloc(&_devPK, 2147483648); //.............................reserva de espaço para chaves privadas
	cudaMemcpyToSymbol(_PK, &_devPK, sizeof(U32*)); //..............init apontador para espaço de chaves privadas

	// reset no device e host (basta uma vez, se altera é porque foi encontrada a solução)
	host_report.final = 0; //.......................................estado inicial do report de finalização => index da chave privada
	host_report.found = 0; //.......................................indicador de finalização
	cudaMemcpyToSymbol(_reportBase, &host_report, sizeof(ReportBase), cudaMemcpyHostToDevice); //passagem para gpu

	// Dados de teste
	//---------------
	U32 hash[8];
	hash[0] = dist(gen);
	hash[1] = dist(gen);
	hash[2] = dist(gen);
	hash[3] = dist(gen);
	hash[4] = dist(gen);
	hash[5] = dist(gen);
	hash[6] = dist(gen);
	hash[7] = dist(gen);
	//Print do h objetivo
	printf("\nHash objetivo => 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X\n",
		hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7]);

	cudaMemcpyToSymbol(hobj, hash, 32, 0, cudaMemcpyHostToDevice);




	// Fill de _PK[] a partir de MSTATE
	setKeys << <512, 512 >> > ();
	cudaDeviceSynchronize();


}

int main(int argc, char** argv) {



	//Inicialização: 
	init();

	U32 m[8] = { 0 };

	const int CICLOS = 32;                                  //32 ciclos => depois PARE OBRIGATORIAMENTE
	const auto durCiclo = std::chrono::minutes(5);          //cada ciclo dura 5 min

	printf("\n %d ciclos de 5 min (total maximo: %d min)\n", CICLOS, CICLOS * 5);

	//******************************************************************************
	//32 CICLOS DE 5 MIN => PASSADO O 32.º CICLO O PROGRAMA PARA OBRIGATORIAMENTE
	//1 ciclo = 5 min de tentativas (máscara aleatória + updateKeys + computeAllHash + verify + relatório)
	//******************************************************************************
	for (int u = 0; u < CICLOS; u++) {

		globalBestSim = 0; // reset do globalBestSim
		auto t_ciclo = std::chrono::steady_clock::now(); //marca o início deste ciclo (u+1)
		U32 cnt = 0; //tentativas feitas neste ciclo a cada uma tentativa corrresponde 67 milhões de chaves/candidatas por ciclo
	

		printf("\n================ CICLO %d/%d (5 min) ================\n", u + 1, CICLOS);

		while (true) { //********** corre enquanto o ciclo não completar 5 min

			//********** PASSOU 5 MIN NESTE CICLO => O WHILE PARE E PASSA AO CICLO SEGUINTE
			if (std::chrono::steady_clock::now() - t_ciclo >= durCiclo) {
				printf("\n[ciclo %d/%d terminado: %u tentativas]\n", u + 1, CICLOS, cnt);
				break;
			}
			//********** fim do teste de tempo do ciclo

			//aplica mascaras sobre candidatas e passa para PK[]
			fillRandomBits(96, m);
			updateKeys << <512, 512 >> > (m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
			cudaDeviceSynchronize();

			//analiza e testa
			computeAllHash << <512, 512 >> > ();
			cudaDeviceSynchronize();

			verify(); //se encontrou a chave final (sim == 256) => para aqui obrigatoriamente

			//********** RELATÓRIO: chave reportada + o SEU HASH ********************
			U32 bestAntes = globalBestSim;
			getBestReport(); //imprime a chave, os bits diferentes e o HASH (SHA256) dessa chave
			if (globalBestSim > bestAntes) { //foi reportada uma nova chave (novo bestsim)
				chavesReportadas++;
				printf("\n[chaves reportadas: %u] ", chavesReportadas);
			}
			//************************************************************************fim do relatório

			cnt++; //incrementa o contador de tentativas deste ciclo
		}
		//************************************************************************fim de ciclo (u+1)
	}

	// ================================================================================
	//32 CICLOS DE 5 MIN CONCLUÍDOS => PARAGEM OBRIGATÓRIA
	// ================================================================================
	printf("\n\n************** FIM: 32 CICLOS DE 5 MIN CONCLUÍDOS => A PARAR O PROGRAMA **************");

	// Apresenta o HASH OBJETIVO (hobj) — o hash usado para calcular a semelhança (sim)
	U32 hobj_h[8];
	cudaMemcpyFromSymbol(hobj_h, hobj, 32, 0, cudaMemcpyDeviceToHost);
	printf("\nHash objetivo (usado no cálculo do sim) => 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X",
		hobj_h[0], hobj_h[1], hobj_h[2], hobj_h[3], hobj_h[4], hobj_h[5], hobj_h[6], hobj_h[7]);

	// Apresenta a melhor chave global (bestsim histórico) + a sua semelhança
	printf("\nMelhor semelhança global: %u bits (%.2f%%) em %u chave(s) reportada(s)",
		globalBestSim, (double)globalBestSim / 256.0 * 100.0, chavesReportadas);
	printf("\nMelhor chave => 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X, 0x%.8X",
		globalBestKey[0], globalBestKey[1], globalBestKey[2], globalBestKey[3],
		globalBestKey[4], globalBestKey[5], globalBestKey[6], globalBestKey[7]);
	printf("\n\n*** FIM — A PARAR O PROGRAMA ***\n");

	system("pause");
	return 0;
}