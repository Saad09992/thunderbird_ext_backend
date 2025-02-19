import { Router } from "express";
import { getCount, processEmails } from "../controller/LangController.js";

const langRouter = Router();

langRouter.get("/api/count", getCount);
langRouter.post("/api/process-emails", processEmails);

export default langRouter;
