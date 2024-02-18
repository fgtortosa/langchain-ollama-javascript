import { Ollama } from "@langchain/community/llms/ollama";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import "@tensorflow/tfjs-node";
import { TensorFlowEmbeddings } from "langchain/embeddings/tensorflow";
import { RetrievalQAChain } from "langchain/chains";

// OLLAMA
const ollama = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "paco-mistral",
});

// DATOS DE LA WEB
const loader = new CheerioWebBaseLoader("https://ceice.gva.es/documents/161863209/369828005/Graus+amb+condicions+especials+en+la+Preinscripció+2023+11+abril+2023.pdf/8ce8b93b-ce86-c74b-eba7-49e3f16c4745?t=1681210184393");
const data = await loader.load();

console.log("Carga de datos de la web");

// TEXTSPLITTER
// Split the text into 500 character chunks. And overlap each chunk by 20 characters
const textSplitter = new RecursiveCharacterTextSplitter({
 chunkSize: 500,
 chunkOverlap: 20
});
const splitDocs = await textSplitter.splitDocuments(data);

console.log("Partiendo en trozos la informacion")

// VECTOR
// Then use the TensorFlow Embedding to store these chunks in the datastore
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, new TensorFlowEmbeddings());

console.log("Vectorizando la información")

// RETRIEVER
const retriever = vectorStore.asRetriever();
const chain = RetrievalQAChain.fromLLM(ollama, retriever);
let result = await chain.call({query: "Me quiero matricular en arquitectuta tecnica. ¿Cuales son los requisitos?"});
console.log("Pregunta:\n: Respuesta:", result.text)
//result = await chain.call({query: "He realizado la prueba de mayores de 45 años en una universidad publica de la comunidad de madrird, ¿me puedo matricular en la universidad de alicante?"});
//console.log("He realizado la prueba de mayores de 45 años en una universidad publica de la comunidad de madrird, ¿me puedo matricular en la universidad de alicante?. Respuesta:\n", result.text)
