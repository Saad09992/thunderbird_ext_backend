import { Router } from "express";
import {
  processEmails,
  registerSession,
  getDataFromFile,
  uploadWritingStyle,
  uploadDataset,
} from "../controller/LangController.js";
const langRouter = Router();

langRouter.post("/api/process-emails", processEmails);
langRouter.post("/api/register-session", registerSession);
langRouter.post("/api/get-data-from-file", getDataFromFile);
langRouter.post("/api/upload-writing-style", uploadWritingStyle);
langRouter.post("/api/upload-dataset",uploadDataset);

export default langRouter;
