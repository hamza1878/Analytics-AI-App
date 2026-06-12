const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageOrientation, LevelFormat, PageBreak
} = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

const COLORS = {
  sprint0: "5B2C8D", sprint1: "1A5276", sprint2: "1E8449",
  sprint3: "B7950B", sprint4: "922B21", sprint5: "1F618D",
  sprint6: "117A65", sprint7: "6E2F9A", sprint8: "784212",
  haute: "E74C3C", moyenne: "F39C12", basse: "27AE60",
  header: "2C3E50", white: "FFFFFF", lightGray: "F2F3F4",
  darkText: "1C1C1C"
};

function hCell(text, widthDXA, bg = COLORS.header, bold = true, size = 18, color = "FFFFFF") {
  return new TableCell({
    borders,
    width: { size: widthDXA, type: WidthType.DXA },
    shading: { fill: bg, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold, size, color, font: "Arial" })]
    })]
  });
}

function dCell(text, widthDXA, bg = "FFFFFF", bold = false, color = COLORS.darkText, center = false) {
  return new TableCell({
    borders,
    width: { size: widthDXA, type: WidthType.DXA },
    shading: { fill: bg, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text: String(text), bold, size: 16, color, font: "Arial" })]
    })]
  });
}

// Sprint data
const sprints = [
  {
    id: "Sprint 0", release: "Foundation v0.0", color: COLORS.sprint0, duration: "1 semaine", points: 40,
    rows: [
      ["Architecture","US-0.1","En tant qu'architecte, je veux définir l'architecture microservices (API Gateway, services, BDD) pour garantir la scalabilité.","Haute",8],
      ["Architecture","US-0.2","En tant que lead dev, je veux configurer les environnements de développement (Git flow, .env templates, scripts d'installation) pour standardiser le travail d'équipe sans containerisation.","Haute",5],
      ["Base de données","US-0.3","En tant qu'architecte, je veux concevoir le modèle de données (ERD, schéma BDD) pour les entités principales (users, rides, vehicles).","Haute",8],
      ["Stack","US-0.4","En tant que développeur, je veux mettre en place la stack technique (NestJS, Flutter, React, PostgreSQL, Redis).","Haute",8],
      ["Sécurité","US-0.5","En tant que développeur, je veux implémenter la configuration sécurité de base (CORS, Helmet, variables d'environnement).","Haute",5],
      ["Qualité","US-0.6","En tant que développeur, je veux créer les squelettes des projets avec structure de dossiers, linting et conventions de code.","Moyenne",3],
      ["Documentation","US-0.7","En tant que développeur, je veux documenter l'API initiale (Swagger/OpenAPI) et les contrats d'interface entre services.","Moyenne",3],
    ]
  },
  {
    id: "Sprint 1", release: "Release v1.0", color: COLORS.sprint1, duration: "2 semaines", points: 43,
    rows: [
      ["Inscription & Connexion","1.1","En tant que passager, je veux créer un compte (email/MdP ou Google).","Haute",8],
      ["Inscription & Connexion","1.2","En tant qu'utilisateur, je veux me connecter de manière sécurisée et rester authentifié.","Haute",5],
      ["Inscription & Connexion","1.3","En tant que passager, je veux utiliser les passkeys pour simplifier l'accès à mon compte.","Haute",5],
      ["Sécurité du compte","1.4","En tant qu'utilisateur, je veux réinitialiser ou changer mon mot de passe.","Haute",5],
      ["Sécurité du compte","1.5","En tant qu'utilisateur, je veux utiliser la biométrie pour valider mes actions sensibles.","Haute",5],
      ["Sécurité du compte","1.6","En tant qu'utilisateur, je veux activer la 2FA (email OTP ou TOTP).","Haute",5],
      ["Gestion des sessions","1.7","En tant qu'utilisateur, je veux consulter et révoquer mes sessions actives.","Moyenne",5],
      ["Gestion des sessions","1.8","En tant qu'utilisateur, je veux supprimer définitivement mon compte.","Haute",5],
    ]
  },
  {
    id: "Sprint 2", release: "Release v1.0", color: COLORS.sprint2, duration: "2 semaines", points: 57,
    rows: [
      ["Gestion utilisateurs","2.1","En tant qu'administrateur, je veux ajouter, modifier, bloquer ou supprimer des comptes.","Haute",8],
      ["Gestion utilisateurs","2.2","En tant qu'utilisateur, je veux consulter et modifier mes informations personnelles.","Moyenne",5],
      ["Gestion des véhicules","2.3","En tant qu'administrateur, je veux créer, modifier, consulter et supprimer des véhicules.","Haute",8],
      ["Gestion des véhicules","2.4","En tant qu'administrateur, je veux gérer l'état des véhicules (disponible, en maintenance).","Haute",5],
      ["Gestion des véhicules","2.5","En tant qu'administrateur, je veux assigner un véhicule à un chauffeur.","Haute",5],
      ["Gestion des classes","2.6","En tant qu'administrateur, je veux créer, modifier, consulter et supprimer des classes de véhicules.","Haute",8],
      ["Gestion des classes","2.7","En tant qu'administrateur, je veux définir les catégories (Economy, Standard, VIP) et y assigner des véhicules.","Moyenne",5],
      ["Zones de travail","2.8","En tant qu'administrateur, je veux créer, modifier, consulter et supprimer des zones de travail.","Haute",8],
      ["Zones de travail","2.10","En tant qu'administrateur, je veux assigner une zone de travail à un chauffeur.","Haute",5],
    ]
  },
  {
    id: "Sprint 3", release: "Release v2.0", color: COLORS.sprint3, duration: "2 semaines", points: 29,
    rows: [
      ["Recherche & Tarification","3.1","En tant que passager, je veux rechercher une destination (texte, voix ou carte) pour définir mon trajet.","Haute",8],
      ["Recherche & Tarification","3.2","En tant que système ML, je veux estimer le prix selon la distance, le trafic et la demande.","Haute",8],
      ["Réservation & Disponibilité","3.3","En tant que passager, je veux consulter les classes disponibles avec leurs prix estimés.","Haute",5],
      ["Réservation & Disponibilité","3.4","En tant que passager, je veux confirmer ma réservation et choisir mon mode de paiement.","Haute",5],
      ["Réservation & Disponibilité","3.5","En tant que chauffeur, je veux gérer ma disponibilité (en ligne / hors ligne).","Haute",3],
    ]
  },
  {
    id: "Sprint 4", release: "Release v2.0", color: COLORS.sprint4, duration: "2 semaines", points: 36,
    rows: [
      ["Attribution de la course","4.1","En tant que système, je veux assigner un chauffeur selon disponibilité, classe et zone.","Haute",8],
      ["Attribution de la course","4.2","En tant que chauffeur, je veux recevoir et visualiser les détails d'une course pour l'accepter ou la refuser.","Haute",5],
      ["Suivi en temps réel","4.3","En tant que passager/chauffeur, je veux suivre la position en temps réel sur la carte.","Haute",8],
      ["Suivi en temps réel","4.4","En tant que chauffeur, je veux mettre à jour le statut de ma course (arrivé, démarrée, terminée).","Haute",5],
      ["Communication & Évaluation","4.5","En tant que passager/chauffeur, je veux communiquer via chat après l'assignation.","Moyenne",5],
      ["Communication & Évaluation","4.6","En tant que passager/chauffeur, je veux évaluer la course et laisser un avis.","Moyenne",5],
    ]
  },
  {
    id: "Sprint 5", release: "Release v2.0", color: COLORS.sprint5, duration: "2 semaines", points: 34,
    rows: [
      ["Assistance libre-service","5.1","En tant que passager, je veux consulter un Help Center pour résoudre mes problèmes.","Moyenne",5],
      ["Assistance libre-service","5.2","En tant qu'utilisateur, je veux utiliser un chatbot IA pour obtenir des réponses 24/7.","Moyenne",8],
      ["Support humain","5.3","En tant que passager, je veux créer un ticket support et discuter en temps réel.","Haute",8],
      ["Support humain","5.4","En tant que passager, je veux déclencher un appel urgent en cas de problème critique.","Haute",5],
      ["Support humain","5.5","En tant qu'administrateur, je veux superviser les tickets support.","Haute",8],
    ]
  },
  {
    id: "Sprint 6", release: "Release v3.0", color: COLORS.sprint6, duration: "2 semaines", points: 44,
    rows: [
      ["Commissions & Promos","6.1","En tant qu'administrateur, je veux gérer les commissions et configurer des codes promotionnels.","Haute",13],
      ["Commissions & Promos","6.2","En tant que chauffeur, je veux consulter mes revenus via un tableau de bord mobile.","Haute",8],
      ["Fidélisation & Facturation","6.3","En tant que passager, je veux accumuler des points de fidélité à chaque course.","Moyenne",5],
      ["Fidélisation & Facturation","6.4","En tant que passager, je veux télécharger mes factures de paiement.","Moyenne",5],
      ["Fidélisation & Facturation","6.5","En tant qu'administrateur, je veux consulter les revenus et transactions de la plateforme.","Haute",13],
    ]
  },
  {
    id: "Sprint 7", release: "Release v3.0", color: COLORS.sprint7, duration: "2 semaines", points: 20,
    rows: [
      ["Notifications de course","7.1","En tant que passager, je veux être notifié quand mon chauffeur est attribué.","Haute",5],
      ["Notifications de course","7.2","En tant qu'utilisateur, je veux recevoir une notification lors de tout changement de statut.","Haute",5],
      ["Notifications financières","7.3","En tant qu'utilisateur, je veux recevoir un email de reçu après chaque paiement.","Moyenne",5],
      ["Notifications financières","7.4","En tant que passager, je veux être notifié en cas d'annulation et de remboursement.","Haute",5],
    ]
  },
  {
    id: "Sprint 8", release: "Release v4.0", color: COLORS.sprint8, duration: "2 semaines", points: 39,
    rows: [
      ["Dashboard & Rapports","8.1","En tant qu'administrateur, je veux accéder à un dashboard analytics de l'activité globale.","Haute",13],
      ["Dashboard & Rapports","8.2","En tant qu'administrateur, je veux recevoir des rapports automatiques sur des périodes définies.","Moyenne",5],
      ["Dashboard & Rapports","8.3","En tant qu'administrateur, je veux exporter les rapports en PDF ou CSV.","Moyenne",5],
      ["Optimisation & IA","8.4","En tant qu'administrateur, je veux analyser les données comportementales pour optimiser les performances.","Moyenne",8],
      ["Optimisation & IA","8.5","En tant qu'administrateur, je veux obtenir des insights IA pour prendre des décisions stratégiques.","Moyenne",8],
    ]
  },
];

function priColor(p) {
  if (p === "Haute") return COLORS.haute;
  if (p === "Moyenne") return COLORS.moyenne;
  return COLORS.basse;
}

function sprintSection(sprint) {
  const sections = [];

  // Sprint header paragraph
  sections.push(new Paragraph({
    spacing: { before: 300, after: 100 },
    children: [
      new TextRun({ text: `${sprint.id}  —  ${sprint.release}`, bold: true, size: 26, color: sprint.color, font: "Arial" }),
      new TextRun({ text: `   |   Durée : ${sprint.duration}   |   Total : ${sprint.points} pts`, size: 18, color: "666666", font: "Arial" }),
    ]
  }));

  // Table with all rows
  const colWidths = [1400, 700, 4200, 900, 700, 900]; // Fonctionnalité, ID, Story, Priorité, Pts, Statut
  const totalW = colWidths.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({
    tableHeader: true,
    children: [
      hCell("Fonctionnalité", colWidths[0]),
      hCell("ID", colWidths[1]),
      hCell("User Story", colWidths[2]),
      hCell("Priorité", colWidths[3]),
      hCell("Pts", colWidths[4]),
      hCell("Statut", colWidths[5]),
    ]
  });

  const dataRows = sprint.rows.map((r, idx) => {
    const [feature, id, story, priority, pts] = r;
    const bg = idx % 2 === 0 ? "FFFFFF" : "F8F9FA";
    const pColor = priColor(priority);
    return new TableRow({
      children: [
        dCell(feature, colWidths[0], bg, false, "374151"),
        dCell(id, colWidths[1], bg, true, sprint.color, true),
        dCell(story, colWidths[2], bg, false, COLORS.darkText),
        new TableCell({
          borders,
          width: { size: colWidths[3], type: WidthType.DXA },
          shading: { fill: pColor, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 80, right: 80 },
          verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: priority, bold: true, size: 14, color: "FFFFFF", font: "Arial" })]
          })]
        }),
        dCell(String(pts), colWidths[4], bg, true, COLORS.darkText, true),
        new TableCell({
          borders,
          width: { size: colWidths[5], type: WidthType.DXA },
          shading: { fill: "FFF9C4", type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 80, right: 80 },
          verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "TODO", bold: false, size: 14, color: "7D6608", font: "Arial" })]
          })]
        }),
      ]
    });
  });

  sections.push(new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  }));

  sections.push(new Paragraph({ spacing: { before: 60, after: 60 }, children: [] }));
  return sections;
}

// Summary table
function summaryTable() {
  const colW = [1200, 1400, 1200, 900, 900];
  const totalW = colW.reduce((a, b) => a + b, 0);
  const hRow = new TableRow({
    tableHeader: true,
    children: [
      hCell("Sprint", colW[0]),
      hCell("Release", colW[1]),
      hCell("Durée", colW[2]),
      hCell("US", colW[3]),
      hCell("Points", colW[4]),
    ]
  });
  const sRows = sprints.map((s, i) => {
    const bg = i % 2 === 0 ? "FFFFFF" : "F2F3F4";
    return new TableRow({ children: [
      new TableCell({ borders, width: { size: colW[0], type: WidthType.DXA }, shading: { fill: s.color, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: s.id, bold: true, size: 16, color: "FFFFFF", font: "Arial" })] })]
      }),
      dCell(s.release, colW[1], bg, false, "374151", true),
      dCell(s.duration, colW[2], bg, false, "374151", true),
      dCell(String(s.rows.length), colW[3], bg, true, COLORS.darkText, true),
      dCell(String(s.points), colW[4], bg, true, COLORS.darkText, true),
    ]});
  });
  const totalUS = sprints.reduce((a, s) => a + s.rows.length, 0);
  const totalPts = sprints.reduce((a, s) => a + s.points, 0);
  const totalRow = new TableRow({ children: [
    hCell("TOTAL", colW[0]),
    hCell("", colW[1]),
    hCell("", colW[2]),
    hCell(String(totalUS), colW[3]),
    hCell(String(totalPts), colW[4]),
  ]});
  return new Table({ width: { size: totalW, type: WidthType.DXA }, columnWidths: colW, rows: [hRow, ...sRows, totalRow] });
}

const children = [
  new Paragraph({
    spacing: { before: 0, after: 200 },
    children: [new TextRun({ text: "Product Backlog — Moviroo", bold: true, size: 40, color: COLORS.header, font: "Arial" })]
  }),
  new Paragraph({
    spacing: { before: 0, after: 400 },
    children: [new TextRun({ text: "Backlog complet incluant Sprint 0 (infrastructure) · 9 sprints · 4 releases", size: 18, color: "666666", font: "Arial", italics: true })]
  }),
  new Paragraph({
    spacing: { before: 0, after: 160 },
    children: [new TextRun({ text: "Résumé des Sprints", bold: true, size: 28, color: COLORS.header, font: "Arial" })]
  }),
  summaryTable(),
  new Paragraph({ spacing: { before: 400, after: 0 }, children: [new PageBreak()] }),
  new Paragraph({
    spacing: { before: 0, after: 200 },
    children: [new TextRun({ text: "Détail du Backlog par Sprint", bold: true, size: 32, color: COLORS.header, font: "Arial" })]
  }),
  ...sprints.flatMap(s => sprintSection(s))
];

const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 15840, height: 12240, orientation: PageOrientation.LANDSCAPE },
        margin: { top: 720, right: 720, bottom: 720, left: 720 }
      }
    },
    children
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/home/claude/product_backlog_moviroo.docx", buf);
  console.log("Done");
});