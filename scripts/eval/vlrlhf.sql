-- MySQL dump 10.13  Distrib 8.0.36, for Linux (x86_64)
--
-- Host: localhost    Database: VLRLHF
-- ------------------------------------------------------
-- Server version	8.0.36-0ubuntu0.22.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Current Database: `VLRLHF`
--

CREATE DATABASE /*!32312 IF NOT EXISTS*/ `VLRLHF` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;

USE `VLRLHF`;

--
-- Table structure for table `HallusionBench`
--

DROP TABLE IF EXISTS `HallusionBench`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `HallusionBench` (
  `tag` varchar(255) NOT NULL,
  `qAcc` float DEFAULT NULL,
  `fAcc` float DEFAULT NULL,
  `easy_aAcc` float DEFAULT NULL,
  `hard_aAcc` float DEFAULT NULL,
  `aAcc` float DEFAULT NULL,
  `PctDiff` float DEFAULT NULL,
  `FPRatio` float DEFAULT NULL,
  `correct` float DEFAULT NULL,
  `inconsistent` float DEFAULT NULL,
  `wrong` float DEFAULT NULL,
  `LH` float DEFAULT NULL,
  `VI` float DEFAULT NULL,
  `Mixed` float DEFAULT NULL,
  PRIMARY KEY (`tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `MME`
--

DROP TABLE IF EXISTS `MME`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `MME` (
  `tag` varchar(255) NOT NULL,
  `existence` float DEFAULT NULL,
  `count` float DEFAULT NULL,
  `position` float DEFAULT NULL,
  `color` float DEFAULT NULL,
  `posters` float DEFAULT NULL,
  `celebrity` float DEFAULT NULL,
  `scene` float DEFAULT NULL,
  `landmark` float DEFAULT NULL,
  `artwork` float DEFAULT NULL,
  `OCR` float DEFAULT NULL,
  `perception` float DEFAULT NULL,
  `commonsense_reasoning` float DEFAULT NULL,
  `numerical_calculation` float DEFAULT NULL,
  `text_translation` float DEFAULT NULL,
  `code_reasoning` float DEFAULT NULL,
  `reasoning` float DEFAULT NULL,
  PRIMARY KEY (`tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `POPE`
--

DROP TABLE IF EXISTS `POPE`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `POPE` (
  `tag` varchar(255) NOT NULL,
  `popular_acc` float DEFAULT NULL,
  `popular_precision` float DEFAULT NULL,
  `popular_recall` float DEFAULT NULL,
  `popular_f1` float DEFAULT NULL,
  `popular_yes_rate` float DEFAULT NULL,
  `adv_acc` float DEFAULT NULL,
  `adv_precision` float DEFAULT NULL,
  `adv_recall` float DEFAULT NULL,
  `adv_f1` float DEFAULT NULL,
  `adv_yes_rate` float DEFAULT NULL,
  `random_acc` float DEFAULT NULL,
  `random_precision` float DEFAULT NULL,
  `random_recall` float DEFAULT NULL,
  `random_f1` float DEFAULT NULL,
  `random_yes_rate` float DEFAULT NULL,
  PRIMARY KEY (`tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `exps`
--

DROP TABLE IF EXISTS `exps`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `exps` (
  `tag` varchar(255) NOT NULL,
  `base_model` varchar(255) DEFAULT NULL,
  `method` varchar(255) DEFAULT NULL,
  `dataset` varchar(255) DEFAULT NULL,
  `epoch` tinyint unsigned DEFAULT '1',
  `step` int unsigned DEFAULT NULL,
  `lr` float DEFAULT NULL,
  `bs` int unsigned DEFAULT NULL,
  PRIMARY KEY (`tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_unicode_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'IGNORE_SPACE,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
/*!50003 CREATE*/ /*!50017 DEFINER=`remote`@`%`*/ /*!50003 TRIGGER `set_bs_on_insert` BEFORE INSERT ON `exps` FOR EACH ROW BEGIN
    IF NEW.base_model LIKE '%llava%' AND NEW.bs IS NULL THEN
        SET NEW.bs = 128;
    END IF;
    IF NEW.base_model LIKE 'QwenVL' AND NEW.bs IS NULL THEN
        SET NEW.bs = 256;
    END IF;
    IF NEW.base_model LIKE 'Silkie' AND NEW.bs IS NULL THEN
        SET NEW.bs = 256;
    END IF;
    IF NEW.base_model LIKE 'internlmxc2vl7b' AND NEW.bs IS NULL THEN
        SET NEW.bs = 128;
    END IF;
    IF NEW.base_model LIKE 'instructblip13b' AND NEW.bs IS NULL THEN
        SET NEW.bs = 128;
    END IF;
END */;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_unicode_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'IGNORE_SPACE,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
/*!50003 CREATE*/ /*!50017 DEFINER=`remote`@`%`*/ /*!50003 TRIGGER `set_lr_on_insert` BEFORE INSERT ON `exps` FOR EACH ROW BEGIN
    IF NEW.base_model LIKE '%llava%' AND NEW.lr IS NULL THEN
        SET NEW.lr = 1e-6;
    END IF;
    IF NEW.base_model LIKE 'QwenVL' AND NEW.lr IS NULL THEN
        SET NEW.lr = 1e-5;
    END IF;
   	IF NEW.base_model LIKE 'Silkie' AND NEW.lr IS NULL THEN
        SET NEW.lr = 1e-5;
    END IF;
    IF NEW.base_model LIKE 'internlmxc2vl7b' AND NEW.lr IS NULL THEN
        SET NEW.lr = 5e-5;
    END IF;
    IF NEW.base_model LIKE 'instructblip13b' AND NEW.lr IS NULL THEN
        SET NEW.lr = 5e-7;
    END IF;
END */;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;

--
-- Table structure for table `mathvista`
--

DROP TABLE IF EXISTS `mathvista`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mathvista` (
  `tag` varchar(255) NOT NULL,
  `Overall` float DEFAULT NULL,
  `scientific_reasoning` float DEFAULT NULL,
  `textbook_question_answering` float DEFAULT NULL,
  `numeric_commonsense` float DEFAULT NULL,
  `arithmetic_reasoning` float DEFAULT NULL,
  `visual_question_answering` float DEFAULT NULL,
  `geometry_reasoning` float DEFAULT NULL,
  `algebraic_reasoning` float DEFAULT NULL,
  `geometry_problem_solving` float DEFAULT NULL,
  `math_word_problem` float DEFAULT NULL,
  `logical_reasoning` float DEFAULT NULL,
  `figure_question_answering` float DEFAULT NULL,
  `statistical_reasoning` float DEFAULT NULL,
  PRIMARY KEY (`tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mmbench`
--

DROP TABLE IF EXISTS `mmbench`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mmbench` (
  `tag` varchar(255) NOT NULL,
  `split` varchar(255) DEFAULT NULL,
  `Overall` float DEFAULT NULL,
  `AR` float DEFAULT NULL,
  `CP` float DEFAULT NULL,
  `FP_C` float DEFAULT NULL,
  `FP_S` float DEFAULT NULL,
  `LR` float DEFAULT NULL,
  `RR` float DEFAULT NULL,
  `action_recognition` float DEFAULT NULL,
  `attribute_comparison` float DEFAULT NULL,
  `attribute_recognition` float DEFAULT NULL,
  `celebrity_recognition` float DEFAULT NULL,
  `function_reasoning` float DEFAULT NULL,
  `future_prediction` float DEFAULT NULL,
  `identity_reasoning` float DEFAULT NULL,
  `image_emotion` float DEFAULT NULL,
  `image_quality` float DEFAULT NULL,
  `image_scene` float DEFAULT NULL,
  `image_style` float DEFAULT NULL,
  `image_topic` float DEFAULT NULL,
  `nature_relation` float DEFAULT NULL,
  `object_localization` float DEFAULT NULL,
  `ocr` float DEFAULT NULL,
  `physical_property_reasoning` float DEFAULT NULL,
  `physical_relation` float DEFAULT NULL,
  `social_relation` float DEFAULT NULL,
  `spatial_relationship` float DEFAULT NULL,
  `structuralized_imagetext_understanding` float DEFAULT NULL,
  PRIMARY KEY (`tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mmmu_dev`
--

DROP TABLE IF EXISTS `mmmu_dev`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mmmu_dev` (
  `tag` varchar(255) NOT NULL,
  `Overall` float DEFAULT NULL,
  `Accounting` float DEFAULT NULL,
  `Agriculture` float DEFAULT NULL,
  `Architecture_and_Engineering` float DEFAULT NULL,
  `Art` float DEFAULT NULL,
  `Art_Theory` float DEFAULT NULL,
  `Basic_Medical_Science` float DEFAULT NULL,
  `Biology` float DEFAULT NULL,
  `Chemistry` float DEFAULT NULL,
  `Clinical_Medicine` float DEFAULT NULL,
  `Computer_Science` float DEFAULT NULL,
  `Design` float DEFAULT NULL,
  `Diagnostics_and_Laboratory_Medicine` float DEFAULT NULL,
  `Economics` float DEFAULT NULL,
  `Electronics` float DEFAULT NULL,
  `Energy_and_Power` float DEFAULT NULL,
  `Finance` float DEFAULT NULL,
  `Geography` float DEFAULT NULL,
  `History` float DEFAULT NULL,
  `Literature` float DEFAULT NULL,
  `Manage` float DEFAULT NULL,
  `Marketing` float DEFAULT NULL,
  `Materials` float DEFAULT NULL,
  `Math` float DEFAULT NULL,
  `Mechanical_Engineering` float DEFAULT NULL,
  `Music` float DEFAULT NULL,
  `Pharmacy` float DEFAULT NULL,
  `Physics` float DEFAULT NULL,
  `Psychology` float DEFAULT NULL,
  `Public_Health` float DEFAULT NULL,
  `Sociology` float DEFAULT NULL,
  `Art_Design` float DEFAULT NULL,
  `Business` float DEFAULT NULL,
  `Health_Medicine` float DEFAULT NULL,
  `Humanities_SocialScience` float DEFAULT NULL,
  `Science` float DEFAULT NULL,
  `Tech_Engineering` float DEFAULT NULL,
  PRIMARY KEY (`tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mmmu_validation`
--

DROP TABLE IF EXISTS `mmmu_validation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mmmu_validation` (
  `tag` varchar(255) NOT NULL,
  `Overall` float DEFAULT NULL,
  `Accounting` float DEFAULT NULL,
  `Agriculture` float DEFAULT NULL,
  `Architecture_and_Engineering` float DEFAULT NULL,
  `Art` float DEFAULT NULL,
  `Art_Theory` float DEFAULT NULL,
  `Basic_Medical_Science` float DEFAULT NULL,
  `Biology` float DEFAULT NULL,
  `Chemistry` float DEFAULT NULL,
  `Clinical_Medicine` float DEFAULT NULL,
  `Computer_Science` float DEFAULT NULL,
  `Design` float DEFAULT NULL,
  `Diagnostics_and_Laboratory_Medicine` float DEFAULT NULL,
  `Economics` float DEFAULT NULL,
  `Electronics` float DEFAULT NULL,
  `Energy_and_Power` float DEFAULT NULL,
  `Finance` float DEFAULT NULL,
  `Geography` float DEFAULT NULL,
  `History` float DEFAULT NULL,
  `Literature` float DEFAULT NULL,
  `Manage` float DEFAULT NULL,
  `Marketing` float DEFAULT NULL,
  `Materials` float DEFAULT NULL,
  `Math` float DEFAULT NULL,
  `Mechanical_Engineering` float DEFAULT NULL,
  `Music` float DEFAULT NULL,
  `Pharmacy` float DEFAULT NULL,
  `Physics` float DEFAULT NULL,
  `Psychology` float DEFAULT NULL,
  `Public_Health` float DEFAULT NULL,
  `Sociology` float DEFAULT NULL,
  `Art_Design` float DEFAULT NULL,
  `Business` float DEFAULT NULL,
  `Health_Medicine` float DEFAULT NULL,
  `Humanities_SocialScience` float DEFAULT NULL,
  `Science` float DEFAULT NULL,
  `Tech_Engineering` float DEFAULT NULL,
  PRIMARY KEY (`tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `mmvet`
--

DROP TABLE IF EXISTS `mmvet`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `mmvet` (
  `tag` varchar(255) NOT NULL,
  `rec` float DEFAULT NULL,
  `ocr` float DEFAULT NULL,
  `know` float DEFAULT NULL,
  `gen` float DEFAULT NULL,
  `spat` float DEFAULT NULL,
  `math` float DEFAULT NULL,
  `total` float DEFAULT NULL,
  PRIMARY KEY (`tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `seedbench`
--

DROP TABLE IF EXISTS `seedbench`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `seedbench` (
  `tag` varchar(255) NOT NULL,
  `SceneUnderstanding` float DEFAULT NULL,
  `InstanceIdentity` float DEFAULT NULL,
  `InstanceAttributes` float DEFAULT NULL,
  `InstanceLocation` float DEFAULT NULL,
  `InstancesCounting` float DEFAULT NULL,
  `SpatialRelation` float DEFAULT NULL,
  `InstanceInteraction` float DEFAULT NULL,
  `VisualReasoning` float DEFAULT NULL,
  `TextUnderstanding` float DEFAULT NULL,
  `Total` float DEFAULT NULL,
  PRIMARY KEY (`tag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-05-22 11:32:56
