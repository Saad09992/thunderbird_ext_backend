import { Router } from "express";
import {
  processEmails,
  registerSession,
  getDataFromFile,
  uploadWritingStyle,
  uploadDataset,
  removeDataset,
  clearWritingStyle,
} from "../controller/LangController.js";
import path from "path";
import { fileURLToPath } from "url";
import multer from "multer";

const __filename = fileURLToPath(import.meta.url);
const currentDirname = path.dirname(__filename);
const rootDir = path.resolve(currentDirname, "../");

const imageStorage = multer.diskStorage({
  destination: path.join(rootDir, "/public/datasets"),
  filename: (req, file, cb) => {
    cb(
      null,
      Date.now() + path.extname(file.originalname)
    );
  },
});

const upload = multer({
  storage: imageStorage,
});

const langRouter = Router();

langRouter.post("/api/process-emails", processEmails);
langRouter.post("/api/register-session", registerSession);
langRouter.post("/api/get-data-from-file", getDataFromFile);
langRouter.post("/api/upload-writing-style", uploadWritingStyle);
langRouter.post("/api/upload-dataset",upload.single("datasetFile") ,uploadDataset);
langRouter.post("/api/remove-dataset",removeDataset);
langRouter.post("/api/clear-writing-style", clearWritingStyle);
export default langRouter;
