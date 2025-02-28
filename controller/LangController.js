import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { Graph } from "@langchain/langgraph";
import * as hub from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import cosineSimilarity from "compute-cosine-similarity";

export const processEmails = async (req, res) => {
  try {
    const { email, userEmails } = req.body;
    const pineconeApiKey = process.env.PINECONE_API_KEY;
    const pc = new Pinecone({ apiKey: pineconeApiKey });

    const allSplits = userEmails.map(
      (chunk, idx) =>
        new Document({
          metadata: { source: "userRequest", emailIndex: idx },
          pageContent: chunk,
        })
    );

    const indexName = "sentemail";
    (async () => {
      const existingIndexes = (await pc.listIndexes()).indexes.map(
        (index) => index.name
      );
      if (!existingIndexes.includes(indexName)) {
        await pc.createIndex({
          name: indexName,
          dimension: 3072,
          metric: "cosine",
        });
      }
    })();

    const index = pc.Index(indexName);

    const embeddings = new OpenAIEmbeddings({
      model: "text-embedding-3-large",
    });

    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: index,
    });
    await vectorStore.addDocuments(allSplits);

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

    const retrieve = async (state) => {
      const retrievedDocs = await vectorStore.similaritySearch(
        state.question,
        5
      );
      return { ...state, context: retrievedDocs};
    };

    const generate = async (state) => {
      // Create new GPT instance for each generation
      const llm = new ChatOpenAI({ model: "gpt-4-turbo", temperature: 0.3 });
      
      const docsContent = state.context
        .map((doc) => doc.pageContent)
        .join("\n\n");
      
      const promptTemplate = ChatPromptTemplate.fromTemplate(`
      You are an AI assistant that helps me to write emails.
      Based on my past emails provided as context. I want you to Analyze the following
       - writing style
       - Message tone,
       - sentence structure,
       - Length of the text
       - formality, and commonly used phrases
      considering the above analysis, generate an email reply to the user input. The response should feel as if I personally wrote it. Only show the email response not the other details. Give Response in proper formatting.
      NOTE: If context is not available or empty ignore it and generate a professional email.
      
      Context: {context}
      User Input: {question}
    `);
      
      const messages = await promptTemplate.invoke({
        question: state.question,
        context: docsContent,
      });
      
      const response = await llm.invoke(messages);
      const similarityScores = await calculateSimilarity(
        response.content,
        state.context
      );
      
      // Destroy the llm instance by setting it to null
      // JavaScript garbage collection will handle cleanup
      return {
        ...state,
        answer: response.content,
        similarities: similarityScores,
      };
    };

    const graph = new Graph();
    graph.addNode("retrieve", retrieve);
    graph.addNode("generate", generate);

    graph.addEdge("__start__", "retrieve");
    graph.addEdge("retrieve", "generate");
    graph.addEdge("generate", "__end__");

    const app = graph.compile();

    const runPipeline = async (question, userEmails) => {
      const result = await app.invoke({ 
        question,
        userEmails // Pass userEmails to the initial state
      });
      return res.status(200).json({
        answer: result["answer"],
        similarity: result["similarities"]?.[0]?.weightedSimilarity || null,
      });
    };

    await runPipeline(email, userEmails);
  } catch (e) {
    return res
      .status(400)
      .json({ error: e.message, message: "Failed to process request" });
  }
};

export const getCount = async (req, res) => {
  return res.status(200).json({ count: 1 });
};