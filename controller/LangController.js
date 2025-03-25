import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { Document } from "langchain/document";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { Graph } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import cosineSimilarity from "compute-cosine-similarity";
import crypto from "crypto";
import csv from "csv-parser";
import { Readable } from "stream";
import fs from "fs";
import { promises as fsPromises } from "fs";
import { WebSocketServer } from "ws";
import pdf from "pdf-parse";
import pdfParse from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import csvParser from "csv-parser";
const wss = new WebSocketServer({ port: 8080 });
const outputPath = "../public/output"

async function getFileData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const contentType = response.headers.get("content-type");
        let data;

        if (
            contentType.includes("text/csv") ||
            contentType.includes("application/csv")
        ) {
            data = await response.text();
        } else if (contentType.includes("application/json")) {
            data = await response.json();
        } else if (contentType.includes("text/")) {
            data = await response.text();
        } else if (contentType.includes("application/pdf")) {
            data = await response.arrayBuffer();
        } else {
            data = await response.blob();
        }
        return {
            success: true,
            data: data,
        };
    } catch (error) {
        console.error("Error fetching file:", error);
        return {
            success: false,
            error: error.message,
        };
    }
}

export const getDataFromFile = async (req, res) => {
    try {
        const { url } = req.body;
        const resp = await getFileData(url);
        const data = resp.data;
        return res
            .status(200)
            .json({ message: "Data fetched successfully", data: data });
    } catch (error) {
        console.error("Error fetching file:", error);
        return res.status(400).json({
            success: false,
            error: error.message,
        });
    }
};

export const processEmails = async (req, res) => {
    try {
        const { email, sessionId, conversationHistory } = req.body;
        const pineconeApiKey = process.env.PINECONE_API_KEY;
        const pc = new Pinecone({ apiKey: pineconeApiKey });

        const historySplits = conversationHistory.map(
            (chunk, idx) =>
                new Document({
                    metadata: { source: "conversationHistory", emailIndex: idx },
                    pageContent: chunk,
                })
        );

        const refrenceIndexName = "refrenceemails";
        const refrenceIndex = pc.Index(refrenceIndexName);
        const datasetIndexName = "dataset";
        const datasetIndex = pc.Index(datasetIndexName);

        const embeddings = new OpenAIEmbeddings({
            model: "text-embedding-3-large",
        });

        const refrenceVectorStore = await PineconeStore.fromExistingIndex(
            embeddings,
            {
                pineconeIndex: refrenceIndex,
            }
        );

        const datasetVectorStore = await PineconeStore.fromExistingIndex(
            embeddings,
            {
                pineconeIndex: datasetIndex,
            }
        );
        const getEmbedding = async (text) => embeddings.embedQuery(text);

        const calculateSimilarity = async (generatedText, retrievedDocs) => {
            if (!retrievedDocs.length) return null;
            const generatedEmbedding = await getEmbedding(generatedText);
            const similarities = await Promise.all(
                retrievedDocs.map(async (doc, idx) => {
                    const docEmbedding = await getEmbedding(doc.pageContent);
                    const similarity = cosineSimilarity(generatedEmbedding, docEmbedding);
                    const weightedSimilarity = similarity / (idx + 1);
                    return { content: doc.pageContent, weightedSimilarity };
                })
            );

            return similarities.sort(
                (a, b) => b.weightedSimilarity - a.weightedSimilarity
            );
        };

        const refrenceEmailsRetrieve = async (state) => {
            const retrievedDocs = await refrenceVectorStore.similaritySearch(
                state.question,
                5,
                { namespace: sessionId }
            );
            return { ...state, refrenceEmails: retrievedDocs };
        };

        const datasetRetrieve = async (state) => {
            const retrievedDocs = await datasetVectorStore.similaritySearch(
                state.question,
                5,
                { namespace: sessionId }
            );
            return { ...state, dataset: retrievedDocs };
        };

        const generateRaw = async (state) => {
            const llm = new ChatOpenAI({ model: "gpt-4-turbo", temperature: 0.3 });

            const docsContent = state.dataset
                .map((doc) => doc.pageContent)
                .join("\n\n");

            const contextGenerationTemplate = ChatPromptTemplate.fromTemplate(`
      You are an AI ass istant that helps me to write emails.
      Based on my past emails provided as context/History. I want you to write a reply to the user based on the Context/History of conversations with the user previously.Make sure to proceed with the discussion based on previous discussions. 
      generate an email reply to the user input. Only show the email response not the other details. Give Response in proper formatting.
      Try to relate the answers to the QNA provided in the dataset.
      NOTE: If context is not available or empty ignore it and generate a professional email reply.
      
      Dataset: {dataset}
      Context/History: {history}
      User Input: {question}
    `);

            const messages = await contextGenerationTemplate.invoke({
                question: state.question,
                history: historySplits,
                dataset: docsContent,
            });

            const response = await llm.invoke(messages);
            // const similarityScores = await calculateSimilarity(
            //   response.content,
            //   state.dataset
            // );

            return {
                ...state,
                answer: response.content,
                // similarities: similarityScores,
            };
        };

        const refineRawGenerate = async (state) => {
            const llm = new ChatOpenAI({ model: "gpt-4-turbo", temperature: 0.3 });

            const docsContent = state.refrenceEmails
                .map((doc) => doc.pageContent)
                .join("\n\n");

            const contextGenerationTemplate = ChatPromptTemplate.fromTemplate(`
      You are an AI assistant that helps me to write emails.
      Based on the refrence emails provided. I want you to refine the email response provided by the user.
      What REFINE mean: Modify the email to the extent that its meaning does not change only the tone, writing style and grammar of the email is modified.
      NOTE: Do not change the email structure, meaning or any other details. Just refine the email. Also remove the Subject line is present only return thr body and last part of emails in short exclude the subject from the email when returning.

      EMAIL: {email}
      Refrence Emails: {refrence} 
    `);

            const messages = await contextGenerationTemplate.invoke({
                email: state.answer,
                refrence: docsContent,
            });

            const response = await llm.invoke(messages);
            const similarityScores = await calculateSimilarity(
                response.content,
                state.dataset
            );

            return {
                ...state,
                answer: response.content,
                similarities: similarityScores,
            };
        };
        const graph = new Graph();
        graph.addNode("datasetRetrieve", datasetRetrieve);
        graph.addNode("generateRaw", generateRaw);
        graph.addNode("refrenceEmailsRetrieve", refrenceEmailsRetrieve);
        graph.addNode("refineRawGenerate", refineRawGenerate);

        graph.addEdge("__start__", "datasetRetrieve");
        graph.addEdge("datasetRetrieve", "generateRaw");
        graph.addEdge("generateRaw", "refrenceEmailsRetrieve");
        graph.addEdge("refrenceEmailsRetrieve", "refineRawGenerate");
        graph.addEdge("refineRawGenerate", "__end__");
        const app = graph.compile();

        const runPipeline = async (question, conversationHistory) => {
            const result = await app.invoke({
                question,
                conversationHistory,
            });
            return res.status(200).json({
                answer: result["answer"],
                similarity: result["similarities"]?.[0]?.weightedSimilarity || null,
            });
        };

        await runPipeline(email, conversationHistory);
    } catch (e) {
        return res
            .status(400)
            .json({ error: e.message, message: "Failed to process request" });
    }
};

export const registerSession = async (req, res) => {
    try {
        const sessionId = crypto.randomUUID();
        return res.status(200).json({
            message: "Successfully registered session",
            sessionId: sessionId,
        });
    } catch (error) {
        return res
            .status(400)
            .json({ message: "Uncatchable Error", error: error.message });
    }
};

export const uploadWritingStyle = async (req, res) => {
    try {
        const { writingStyle, sessionId } = req.body;
        const pineconeApiKey = process.env.PINECONE_API_KEY;
        const pc = new Pinecone({ apiKey: pineconeApiKey });

        const embeddings = new OpenAIEmbeddings({
            model: "text-embedding-3-large",
        });

        let completed = 0;
        const total = writingStyle.length;
        const allSplits = await Promise.all(
            writingStyle.map(async (chunk, idx) => {
                const vector = await embeddings.embedQuery(chunk);
                completed++;
                wss.clients.forEach((client) => {
                    if (client.readyState === 1) {
                        client.send(
                            JSON.stringify({ sessionId, progress: (completed / total) * 100 })
                        );
                    }
                });
                return {
                    id: `${sessionId}-${idx}`,
                    values: vector,
                    metadata: { source: "writingStyle", emailIndex: idx, sessionId },
                };
            })
        );

        const refrenceIndexName = "refrenceemails";
        const existingIndexes = (await pc.listIndexes()).indexes.map(
            (index) => index.name
        );

        if (!existingIndexes.includes(refrenceIndexName)) {
            console.log("Creating reference index...");
            await pc.createIndex({
                name: refrenceIndexName,
                dimension: 3072,
                metric: "cosine",
                spec: {
                    serverless: {
                        cloud: "aws",
                        region: "us-east-1",
                    },
                },
            });
        }

        const refrenceIndex = pc.Index(refrenceIndexName);
        const stats = await refrenceIndex.describeIndexStats();
        if (
            stats.namespaces &&
            stats.namespaces[sessionId] &&
            stats.namespaces[sessionId].vectorCount > 0
        ) {
            try {
                await refrenceIndex.namespace(sessionId).deleteAll();
            } catch (error) {
                console.log("Namespace:", sessionId, "Cannot be deleted");
            }
        }
        await refrenceIndex.namespace(sessionId).upsert(allSplits);
        return res.status(200).json({ message: "Data received successfully" });
    } catch (error) {
        console.log(error);
        return res
            .status(400)
            .json({ message: "Uncatchable Error", error: error.message });
    }
};

export const uploadDataset = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ message: "No file found" });
        }

        const { sessionId } = req.body;
        console.log("SessionId: ", sessionId);

        const filePath = req.file.path;
        const fileExt = req.file.mimetype;
        const pineconeApiKey = process.env.PINECONE_API_KEY;

        const pc = new Pinecone({ apiKey: pineconeApiKey });
        const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-large" });

        // Convert file to text
        const textContent = await fileToTxt(filePath, fileExt);

        // Split text into chunks
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 300,
            chunkOverlap: 30,
        });
        const processedDataset = await textSplitter.splitText(textContent);

        console.log("Generating embeddings in BATCH mode...");



        const batchSize = 50;
        let completedEmbedding = 0;
        let completedUploading = 0;
        const total = processedDataset.length;
        const vectors = [];

        // Process embeddings in batches
        for (let i = 0; i < processedDataset.length; i += batchSize) {
            const batch = processedDataset.slice(i, i + batchSize);
            const batchEmbeddings = await embeddings.embedDocuments(batch);

            batchEmbeddings.forEach((vector, idx) => {
                vectors.push({
                    id: `${sessionId}-${i + idx}`,
                    values: vector,
                    metadata: { source: "dataset", emailIndex: i + idx, sessionId },
                });
            });

            completedEmbedding += batch.length;
            sendProgress(sessionId, (completedEmbedding / total) * 100, "embedding");
            console.log(`Batch ${i / batchSize + 1} processed...`);
        }

        console.log("History Splits processed");

        const indexName = "dataset";
        const existingIndexes = (await pc.listIndexes()).indexes.map((index) => index.name);

        if (!existingIndexes.includes(indexName)) {
            console.log("Creating reference index...");
            await pc.createIndex({
                name: indexName,
                dimension: 3072,
                metric: "cosine",
                spec: {
                    serverless: {
                        cloud: "aws",
                        region: "us-east-1",
                    },
                },
            });
        }

        const index = pc.Index(indexName);
        const stats = await index.describeIndexStats();

        if (stats.namespaces && stats.namespaces[sessionId] && stats.namespaces[sessionId].vectorCount > 0) {
            try {
                await index.namespace(sessionId).deleteAll();
            } catch (error) {
                console.log("Namespace:", sessionId, "Cannot be deleted");
            }
        }

        console.log("Saving data to Pinecone...");
        for (let i = 0; i < vectors.length; i += batchSize) {
            const batch = vectors.slice(i, i + batchSize);
            await index.namespace(sessionId).upsert(batch);
            completedUploading += batch.length;
            sendProgress(sessionId, (completedUploading/ total) * 100, "uploading");
            console.log(`Upserted batch ${i / batchSize + 1}`);
        }

        sendProgress(sessionId, 100); // Ensure progress reaches 100% at the end

        return res.status(200).json({ message: "Data uploaded successfully" });

    } catch (error) {
        console.log(error);
        return res.status(400).json({ message: "Error processing dataset", error: error.message });
    }
};

function sendProgress(sessionId, progress, type) {
    wss.clients.forEach((client) => {
        if (client.readyState === 1) {
            client.send(JSON.stringify({ sessionId, progress, type }));
        }
    });
}

export const removeDataset = async (req, res) => {
    try {
        const { sessionId } = req.body;
        const pineconeApiKey = process.env.PINECONE_API_KEY;
        const pc = new Pinecone({ apiKey: pineconeApiKey });
        if (sessionId == "" || sessionId == null) {
            return res.status(400).json({ message: "No session id found" });
        }
        const indexName = "dataset";
        const index = pc.index(indexName);
        const stats = await index.describeIndexStats();
        console.log(stats.namespaces)
        if (
            stats.namespaces &&
            stats.namespaces[sessionId]
        ) {
            try {
                console.log("deleting namespace")
                await index.namespace(sessionId).deleteAll();
            } catch (error) {
                console.log("Namespace:", sessionId, "Cannot be deleted");
               return res.status(400).json({ message: "Uncatchable Error", error: err.message });
            }
        }
        return res.status(200).json({ message: "Dataset removed" });
    } catch (err) {
        return res.status(400).json({ message: "Uncatchable Error", error: err.message });
    }
}

export const clearWritingStyle =async(req,res)=>{
    try {
        const { sessionId } = req.body;
        const pineconeApiKey = process.env.PINECONE_API_KEY;
        const pc = new Pinecone({ apiKey: pineconeApiKey });
        if (sessionId == "" || sessionId == null) {
            return res.status(400).json({ message: "No session id found" });
        }
        const indexName = "refrenceemails";
        const index = pc.index(indexName);
        const stats = await index.describeIndexStats();
        console.log(stats.namespaces)
        if (
            stats.namespaces &&
            stats.namespaces[sessionId]
        ) {
            try {
                console.log("deleting namespace")
                await index.namespace(sessionId).deleteAll();
            } catch (error) {
                console.log("Namespace:", sessionId, "Cannot be deleted");
                return res.status(400).json({ message: "Uncatchable Error", error: err.message });
            }
        }
        return res.status(200).json({ message: "Emails  removed" });
    } catch (err) {
        console.log(err)
        return res.status(400).json({ message: "Uncatchable Error", error: err.message });
    }
}

async function fileToTxt(filePath,ext){
    try {
        if (ext === "application/pdf") {
            const dataBuffer = await fsPromises.readFile(filePath);
            const data = await pdfParse(dataBuffer);
            return data.text;
        } else if (ext === "text/csv") {
            const rows = [];
            return new Promise((resolve, reject) => {
                fs.createReadStream(filePath)
                    .pipe(csvParser())
                    .on("data", (row) => {
                        rows.push(Object.values(row).join(" ")); // Convert CSV row to space-separated text
                    })
                    .on("end", () => {
                        resolve(rows.join("\n")); // Join rows with newline
                    })
                    .on("error", (err) => {
                        reject("Error parsing CSV: " + err);
                    });
            });
        } else {
            throw new Error("Unsupported file type");
        }
    } catch (err) {
        console.error("Error converting file to TXT:", err);
        return "";
    }
}

