% arxiv.cls
% arXiv 风格的LaTeX文档类
% 基于NIPS样式修改

\ProvidesClass{arxiv}[2018/11/18 arXiv style]

% 加载基础类
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[10pt,twoside,twocolumn]{article}

% 数学包
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{mathtools}
\RequirePackage{amsthm}

% 图像
\RequirePackage{graphicx}

% 设置页面大小和边距
\RequirePackage[margin=1in,letterpaper]{geometry}

% 字体
\RequirePackage{times}

% 图片和表格控制
\RequirePackage{caption}
\RequirePackage{subcaption}

% 定制题头空间
\setlength{\headheight}{10pt}
\setlength{\headsep}{10pt}

% 设置段落间距
\setlength{\parskip}{10pt plus 2pt minus 1pt}

% 改进的超链接
\RequirePackage{xcolor}
\RequirePackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=blue,
    citecolor=blue,
    urlcolor=blue,
}

% 重定义标题格式
\renewcommand{\section}{\@startsection{section}{1}{\z@}%
                       {-10pt \@plus -2pt \@minus -1pt}%
                       {6pt \@plus 2pt \@minus 1pt}%
                       {\normalfont\normalsize\bfseries}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
                       {-8pt \@plus -2pt \@minus -1pt}%
                       {4pt \@plus 2pt \@minus 1pt}%
                       {\normalfont\normalsize\bfseries}}
\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}%
                       {-6pt \@plus -2pt \@minus -1pt}%
                       {2pt \@plus 2pt \@minus 1pt}%
                       {\normalfont\normalsize\bfseries}}

% 参考文献格式
\bibliographystyle{plainnat}

\endinput
